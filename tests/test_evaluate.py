"""Unit tests for src/evaluate.py on a small hand-built network (no real data)."""
import json
import numpy as np
import pytest

from src import evaluate as ev

# Straight east-west road, 4 nodes ~100 m apart at lat 50; arcs in both directions.
NODES = np.array([[8.0000, 50.0], [8.0014, 50.0], [8.0028, 50.0], [8.0042, 50.0]])
ARCS = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
ROUTE = np.array([0, 1, 2])  # drives west -> east


def make_trip():
    lons = np.linspace(8.0002, 8.0040, 12)
    lats = np.full_like(lons, 50.00005)  # ~5 m north of the road
    track = np.column_stack([lons, lats, np.arange(12.0)])
    return {"trip_id": "t1", "track": track, "nodes": NODES, "arcs": ARCS, "route": ROUTE}


def test_reference_labels_monotone_and_close():
    ref, dist = ev.derive_reference_labels(make_trip())
    assert list(ref[:2]) == [0, 0] and list(ref[-2:]) == [2, 2]
    assert np.all(np.diff([list(ROUTE).index(a) for a in ref]) >= 0)
    assert np.all(dist < 7)


def test_reference_labels_handle_revisits():
    # out and back on the same road: route 0,1,4,3 ; points go east then west
    lons = np.concatenate([np.linspace(8.0002, 8.0026, 6), np.linspace(8.0026, 8.0002, 6)])
    trip = {"trip_id": "t2", "track": np.column_stack([lons, np.full(12, 50.0), np.arange(12.0)]),
            "nodes": NODES, "arcs": ARCS, "route": np.array([0, 1, 4, 3])}
    ref, _ = ev.derive_reference_labels(trip)
    assert ref[0] == 0 and ref[-1] == 3
    # directed arcs 1 and 4 are geometrically identical; order must put 1 before 4
    assert list(ev.collapse(ref)) in ([0, 1, 4, 3], [0, 1, 3], [0, 4, 3])


def test_accuracy_and_undirected():
    ref = np.array([0, 0, 1, 1])
    pred = np.array([0, 3, 1, 2])  # arc 3 is reverse of 0; arc 2 is a different road
    assert ev.point_accuracy(pred, ref) == 0.5
    assert ev.point_accuracy_undirected(pred, ref, ARCS) == 0.75


def test_continuity():
    assert ev.route_continuity([0, 0, 1, 2], ARCS) == (1.0, 0)
    cont, breaks = ev.route_continuity([0, 2, 2], ARCS)  # 0 -> 2 skips arc 1
    assert breaks == 1 and cont == 0.5


def test_route_set_and_sequence():
    p, r, f = ev.route_arc_prf([0, 0, 1, 5], [0, 1, 2])
    assert (p, r) == (2 / 3, 2 / 3)
    assert ev.collapse([5, 5, 7, 7, 5]) == [5, 7, 5]
    assert ev.levenshtein([1, 2, 3], [1, 3]) == 1
    assert ev.sequence_similarity([0, 0, 1, 1, 2], ROUTE) == 1.0


def test_macro_f1_perfect():
    assert ev.macro_prf([0, 1, 2], [0, 1, 2]) == (1.0, 1.0, 1.0)


def test_candidate_miss_rate():
    assert ev.candidate_miss_rate([[0, 3], [2], [1, 4]], [0, 1, 1]) == pytest.approx(1 / 3)


def test_invalid_predictions_are_flagged():
    trip = make_trip()
    assert ev.score_trip(trip, [0, 1])["status"] == "invalid"
    assert ev.score_trip(trip, [99] * 12)["status"] == "invalid"


def test_end_to_end_with_matcher(tmp_path):
    tdir = tmp_path / "part" / "t1"
    tdir.mkdir(parents=True)
    trip = make_trip()
    np.savetxt(tdir / "t1.track", trip["track"])
    np.savetxt(tdir / "t1.nodes", NODES)
    np.savetxt(tdir / "t1.arcs", ARCS, fmt="%d")
    np.savetxt(tdir / "t1.route", ROUTE, fmt="%d")

    seen = {}
    def always_first_arc(trace, road_network):
        seen["keys"] = sorted(road_network)
        return [0] * len(trace)

    rows, summary = ev.evaluate(tmp_path / "part", matcher=always_first_arc, log=lambda *_: None)
    assert seen["keys"] == ["arcs", "nodes"]  # no ground truth handed to the matcher
    assert rows[0]["status"] == "ok" and 0 < rows[0]["point_accuracy"] < 1
    csv_p, json_p = ev.write_results(rows, summary, tmp_path / "res", "dummy", {"run_name": "dummy"})
    assert csv_p.exists() and json.loads(json_p.read_text())["summary"]["n_trips_ok"] == 1


def test_unmatched_points_count_as_wrong_not_invalid():
    trip = make_trip()
    ref, dist = ev.derive_reference_labels(trip)
    pred = ref.copy()
    pred[:3] = -1                     # 3 of 12 points unmatched
    row = ev.score_trip(trip, pred, ref, dist)
    assert row["status"] == "ok"
    assert row["unmatched_rate"] == 0.25
    assert row["point_accuracy"] == 0.75
    assert row["point_accuracy_undirected"] == 0.75   # -1 must not alias arcs[-1]
    assert row["route_breaks"] >= 1                   # pair (-1, arc) is a break
    assert -1 not in ev.collapse(x for x in pred if x != -1)
    assert ev.score_trip(trip, [-2] * 12, ref, dist)["status"] == "invalid"


def test_results_json_is_strict_json_without_nan(tmp_path):
    rows = [{"trip_id": "t1", "status": "ok", "inference_time_s": float("nan"), "point_accuracy": 1.0}]
    summary = {"inference_time_s_mean": float("nan"), "n": 1}
    _, json_p = ev.write_results(rows, summary, tmp_path, "x", {"run_name": "x"})
    text = json_p.read_text()
    assert "NaN" not in text
    data = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert data["per_trip"][0]["inference_time_s"] is None
