from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


def trivial_matcher(trace, road_network):
    """Choose a valid arc for every synthetic GPS point."""
    return [0 if i < 4 else 1 if i < 7 else 2 for i in range(len(trace))]


@pytest.fixture
def client(tmp_path, monkeypatch):
    trips_dir = tmp_path / "student_test"
    trip_dir = trips_dir / "trip-a"
    trip_dir.mkdir(parents=True)

    nodes = np.array([[0.0, 0.0], [0.001, 0.0], [0.002, 0.0], [0.003, 0.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0]])
    trace = np.column_stack([
        np.linspace(0.00005, 0.00295, 10),
        np.zeros(10),
        np.arange(10, dtype=float),
    ])
    np.savetxt(trip_dir / "trip-a.track", trace)
    np.savetxt(trip_dir / "trip-a.nodes", nodes)
    np.savetxt(trip_dir / "trip-a.arcs", arcs, fmt="%d")
    np.savetxt(trip_dir / "trip-a.route", np.array([0, 1, 2]), fmt="%d")

    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(trips_dir))
    monkeypatch.setenv("PATHSNAP_MATCHER", "test:trivial_matcher")
    monkeypatch.setattr(api_main, "load_matcher", lambda _spec: trivial_matcher)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_and_trips(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model": "test:trivial_matcher"}
    assert client.get("/trips").json() == {"trips": ["trip-a"]}


def test_predict_has_contract_fields_without_ground_truth(client):
    response = client.post("/predict", json={"trip_id": "trip-a"})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "trip_id", "model", "n_points", "gps_points", "predicted_route",
        "metrics", "inference_time_s",
    }
    assert payload["trip_id"] == "trip-a"
    assert payload["n_points"] == 10
    assert len(payload["gps_points"]) == 10
    assert len(payload["predicted_route"]) == 3
    assert set(payload["metrics"]) == {
        "point_accuracy", "route_continuity", "macro_f1", "unmatched_rate",
    }
    assert payload["inference_time_s"] >= 0
    assert "route" not in payload


def test_unknown_trip_returns_404(client):
    response = client.post("/predict", json={"trip_id": "missing"})
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown trip_id"}


def test_matcher_failure_returns_500(client, monkeypatch):
    def fail(_trace, _network):
        raise RuntimeError("matcher exploded")

    monkeypatch.setattr(api_main.app.state, "matcher", fail)
    response = client.post("/predict", json={"trip_id": "trip-a"})
    assert response.status_code == 500
    assert response.json() == {"detail": "matcher exploded"}


def test_request_logging_includes_request_fields(client, caplog):
    with caplog.at_level(logging.INFO, logger="pathsnap.api"):
        client.post("/predict", json={"trip_id": "trip-a"})
    assert "method=POST" in caplog.text
    assert "path=/predict" in caplog.text
    assert "trip_id=trip-a" in caplog.text
    assert "status_code=200" in caplog.text
    assert "duration_s=" in caplog.text
