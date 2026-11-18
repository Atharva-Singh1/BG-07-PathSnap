"""HTTP client and map building for the PathSnap UI (no Streamlit here, so it is testable).

Implements the client side of docs/api_contract.md.
"""

from __future__ import annotations

import time

import requests

REQUIRED_KEYS = {"trip_id", "gps_points", "predicted_route", "metrics"}
METRIC_KEYS = ("point_accuracy", "route_continuity", "macro_f1", "unmatched_rate")
# Anything that looks like ground truth must never reach the UI.
FORBIDDEN_KEYS = {"route", "reference_route", "reference", "ground_truth", "ref_labels", "labels"}


class ApiError(RuntimeError):
    pass


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _get(base: str, path: str, timeout: float) -> dict:
    try:
        r = requests.get(_url(base, path), timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"cannot reach the API at {base}: {exc.__class__.__name__}") from exc
    if r.status_code != 200:
        raise ApiError(f"{path} returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def health(base: str, timeout: float = 5) -> dict:
    return _get(base, "/health", timeout)


def list_trips(base: str, timeout: float = 10) -> list[str]:
    data = _get(base, "/trips", timeout)
    trips = data.get("trips")
    if not isinstance(trips, list) or not trips:
        raise ApiError("/trips returned no trips")
    return [str(t) for t in trips]


def validate_prediction(data: dict) -> dict:
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ApiError(f"/predict response is missing {sorted(missing)}")
    leaked = FORBIDDEN_KEYS & set(data)
    if leaked:
        raise ApiError(f"/predict response contains ground-truth fields {sorted(leaked)}; "
                       "the API must keep these server-side")
    if not data["gps_points"]:
        raise ApiError("/predict returned no GPS points")
    return data


def predict(base: str, trip_id: str, timeout: float = 180) -> tuple[dict, float]:
    """POST /predict. Returns (response, round-trip seconds)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(_url(base, "/predict"), json={"trip_id": trip_id}, timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"cannot reach the API at {base}: {exc.__class__.__name__}") from exc
    rtt = time.perf_counter() - t0
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"/predict returned HTTP {r.status_code}: {str(detail)[:300]}")
    return validate_prediction(r.json()), rtt


def build_map(data: dict):
    """Folium map: GPS points and predicted route, with legend and layer control."""
    import folium

    pts = [(lat, lon) for lon, lat in data["gps_points"]]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                  [max(p[0] for p in pts), max(p[1] for p in pts)]])

    route = folium.FeatureGroup(name="Predicted route", show=True)
    for seg in data["predicted_route"]:
        folium.PolyLine([(lat, lon) for lon, lat in seg], color="#d7191c", weight=5,
                        opacity=0.85).add_to(route)
    route.add_to(m)

    gps = folium.FeatureGroup(name="GPS points", show=True)
    for lat, lon in pts:
        folium.CircleMarker((lat, lon), radius=2, color="#2b83ba", fill=True,
                            fill_opacity=0.9, weight=0).add_to(gps)
    gps.add_to(m)

    folium.Marker(pts[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(pts[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>Legend</b><br>
      <span style="color:#2b83ba;">&#9679;</span> GPS points<br>
      <span style="color:#d7191c;">&#9644;</span> Predicted route<br>
      <span style="color:green;">&#9873;</span> Start &nbsp; <span>&#9873;</span> End
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
