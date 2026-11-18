"""PathSnap evaluator UI (laptop 2).

Run:  streamlit run ui/app.py
The API address is set in the sidebar, or with the PATHSNAP_API_URL environment variable.
The UI only receives predictions and server-computed metrics; ground truth stays on the API.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import METRIC_KEYS, ApiError, build_map, health, list_trips, predict  # noqa: E402

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.title("PathSnap: GPS-to-road map matching")

with st.sidebar:
    st.header("API (laptop 1)")
    api = st.text_input("API URL", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
    if st.button("Check connection"):
        try:
            h = health(api)
            st.success(f"Connected. Model: {h.get('model', 'unknown')}")
        except ApiError as exc:
            st.error(str(exc))
    st.caption("Ground truth stays on the server. This UI only receives the predicted route and "
               "the accuracy the server computed.")


@st.cache_data(ttl=30, show_spinner=False)
def cached_trips(base: str) -> list[str]:
    return list_trips(base)


try:
    trips = cached_trips(api)
except ApiError as exc:
    st.error(f"Could not load the trip list. {exc}")
    st.info("Start the API on laptop 1 and check the URL in the sidebar.")
    st.stop()

col_a, col_b = st.columns([3, 1])
trip = col_a.selectbox(f"Evaluation trip ({len(trips)} available)", trips)
run = col_b.button("Match route", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Matching trip {trip} on the API ..."):
        try:
            data, rtt = predict(api, trip)
            st.session_state["result"] = (data, rtt)
        except ApiError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

if "result" in st.session_state:
    data, rtt = st.session_state["result"]
    m = data["metrics"]
    st.subheader(f"Trip {data['trip_id']}"
                 + (f"  ·  model: {data['model']}" if data.get("model") else ""))
    labels = {"point_accuracy": "Top-1 accuracy", "route_continuity": "Route continuity",
              "macro_f1": "Macro F1", "unmatched_rate": "Unmatched points"}
    cols = st.columns(len(METRIC_KEYS) + 2)
    for c, k in zip(cols, METRIC_KEYS):
        v = m.get(k)
        c.metric(labels[k], "n/a" if v is None else f"{100 * v:.1f}%")
    inf = data.get("inference_time_s")
    cols[-2].metric("Server time", "n/a" if inf is None else f"{inf:.2f} s")
    cols[-1].metric("Round trip", f"{rtt:.2f} s")
    st.caption(f"{data.get('n_points', len(data['gps_points']))} GPS points · "
               f"{len(data['predicted_route'])} predicted arcs. Metrics computed on the API "
               "against server-side reference data.")
    st_folium(build_map(data), height=560, use_container_width=True, returned_objects=[])
    with st.expander("Raw API response (without coordinates)"):
        st.json({k: v for k, v in data.items() if k not in ("gps_points", "predicted_route")})
