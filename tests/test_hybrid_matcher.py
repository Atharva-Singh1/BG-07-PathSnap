"""Tests for src/hybrid_matcher.py on a small hand-built two-way road (no real data)."""
import numpy as np
import pytest

from src import hybrid_matcher
from src.features import FEATURE_NAMES, CandidateConfig
from src.hmm_matcher import match as hmm_match
from src.hybrid_matcher import match, match_with

NODES = np.array([[8.0000, 50.0], [8.0014, 50.0], [8.0028, 50.0], [8.0042, 50.0]])
ARCS = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
NET = {"nodes": NODES, "arcs": ARCS}
EAST, WEST = [0, 1, 2], [3, 4, 5]


def eastbound_trace(n=12):
    lons = np.linspace(8.0002, 8.0040, n)
    return np.column_stack([lons, np.full(n, 50.00005), np.arange(float(n))])


class FakeModel:
    """p = 0.95 for the preferred travel direction (via heading_cos), 0.05 otherwise."""

    def __init__(self, prefer_east=True):
        self.sign = 1.0 if prefer_east else -1.0

    def predict_proba(self, X):
        hc = X[:, FEATURE_NAMES.index("heading_cos")] * self.sign
        p = np.where(hc > 0, 0.95, 0.05)
        return np.column_stack([1 - p, p])


def bundle(prefer_east=True):
    return {"model": FakeModel(prefer_east), "candidate_config": CandidateConfig().to_dict()}


def test_one_arc_id_per_point():
    pred = match_with(eastbound_trace(), NET, weight=1.0, bundle=bundle())
    assert len(pred) == 12
    assert all(isinstance(a, int) for a in pred)
    assert set(pred) <= set(range(len(ARCS)))


def test_weight_zero_equals_hmm_only():
    trace = eastbound_trace()
    assert match_with(trace, NET, weight=0, bundle=bundle()) == hmm_match(trace, NET)


def test_classifier_decides_between_equally_close_directions():
    trace = eastbound_trace()
    assert set(match_with(trace, NET, weight=4, bundle=bundle(prefer_east=True))) <= set(EAST)
    assert set(match_with(trace, NET, weight=4, bundle=bundle(prefer_east=False))) <= set(WEST)


def test_unscored_arcs_get_the_floor_so_result_is_hmm_only(monkeypatch):
    class Empty:
        point = np.array([], dtype=int)
        arc = np.array([], dtype=int)

    monkeypatch.setattr(hybrid_matcher, "candidate_scores",
                        lambda trace, net, bundle=None: (Empty(), np.array([])))
    trace = eastbound_trace()
    assert match_with(trace, NET, weight=2.0, bundle={}) == hmm_match(trace, NET)


def test_match_reads_environment(monkeypatch):
    monkeypatch.setattr(hybrid_matcher, "load_bundle", lambda path: bundle(prefer_east=False))
    monkeypatch.setenv("PATHSNAP_MODEL", "ignored.joblib")
    monkeypatch.setenv("PATHSNAP_CLASSIFIER_WEIGHT", "4")
    assert set(match(eastbound_trace(), NET)) <= set(WEST)
    monkeypatch.setenv("PATHSNAP_CLASSIFIER_WEIGHT", "0")
    assert match(eastbound_trace(), NET) == hmm_match(eastbound_trace(), NET)


def test_default_weight_is_one(monkeypatch):
    monkeypatch.delenv("PATHSNAP_CLASSIFIER_WEIGHT", raising=False)
    assert hybrid_matcher.classifier_weight() == 1.0
