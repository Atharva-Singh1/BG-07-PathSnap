"""Tests for src/features.py on a small hand-built two-way road (no real data)."""
import numpy as np

from src.features import FEATURE_NAMES, build_candidates, label_candidates, top1

# Straight east-west road, 4 nodes ~100 m apart at lat 50, arcs in both directions.
NODES = np.array([[8.0000, 50.0], [8.0014, 50.0], [8.0028, 50.0], [8.0042, 50.0]])
ARCS = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
NET = {"nodes": NODES, "arcs": ARCS}


def eastbound_trace(n=12):
    lons = np.linspace(8.0002, 8.0040, n)
    return np.column_stack([lons, np.full(n, 50.00005), np.arange(float(n))])


def test_every_point_has_candidates_and_features():
    c = build_candidates(eastbound_trace(), NET)
    assert c.X.shape == (len(c.arc), len(FEATURE_NAMES))
    assert set(c.point) == set(range(12))
    assert np.all(np.isfinite(c.X))


def test_heading_separates_the_two_directions():
    c = build_candidates(eastbound_trace(), NET)
    hc = c.X[:, FEATURE_NAMES.index("heading_cos")]
    eastbound = np.isin(c.arc, [0, 1, 2])
    assert np.all(hc[eastbound] > 0.9) and np.all(hc[~eastbound] < -0.9)


def test_far_point_still_gets_nearest_arc():
    trace = np.array([[8.0020, 50.01, 0.0], [8.0021, 50.01, 1.0]])  # ~1.1 km north
    c = build_candidates(trace, NET)
    assert set(c.point) == {0, 1}


def test_labels_and_top1():
    c = build_candidates(eastbound_trace(), NET)
    ref = np.where(np.linspace(8.0002, 8.0040, 12) < 8.0014, 0,
                   np.where(np.linspace(8.0002, 8.0040, 12) < 8.0028, 1, 2))
    y = label_candidates(c, ref)
    assert y.sum() == 12                                  # one correct candidate per point
    pred = top1(c.point, c.arc, y.astype(float), c.n_points)
    assert np.array_equal(pred, ref)
