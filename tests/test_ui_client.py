"""Tests for ui/client.py against ui/mock_api.py (fake data, no dataset, no model)."""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import client  # noqa: E402
import mock_api  # noqa: E402


@pytest.fixture(scope="module")
def api():
    server = mock_api.make_server(0)          # port 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_health_and_trips(api):
    assert client.health(api)["status"] == "ok"
    assert client.list_trips(api) == mock_api.MOCK_TRIPS


def test_predict_roundtrip(api):
    data, rtt = client.predict(api, mock_api.MOCK_TRIPS[0])
    assert data["trip_id"] == mock_api.MOCK_TRIPS[0] and rtt >= 0
    assert set(client.METRIC_KEYS) <= set(data["metrics"])


def test_unknown_trip_is_a_clear_error(api):
    with pytest.raises(client.ApiError, match="404"):
        client.predict(api, "no-such-trip")


def test_unreachable_api_is_a_clear_error():
    with pytest.raises(client.ApiError, match="cannot reach"):
        client.list_trips("http://127.0.0.1:9", timeout=2)


def test_ground_truth_in_response_is_rejected():
    data = mock_api.fake_prediction(mock_api.MOCK_TRIPS[0])
    data["reference_route"] = [[0, 0]]
    with pytest.raises(client.ApiError, match="ground-truth"):
        client.validate_prediction(data)


def test_build_map(api):
    pytest.importorskip("folium")
    data, _ = client.predict(api, mock_api.MOCK_TRIPS[1])
    html = client.build_map(data).get_root().render()
    assert "Predicted route" in html and "GPS points" in html and "Legend" in html
