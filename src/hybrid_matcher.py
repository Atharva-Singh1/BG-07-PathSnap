"""Hybrid matcher: trained candidate classifier + HMM/Viterbi decoder.

The HMM (src/hmm_matcher.py) is unchanged. Its emission hook `emission_extra_fn`
adds, for every GPS point and every HMM candidate arc, a log-score taken from the
classifier's probability that the arc is the correct one:

    extra(i, arc) = weight * log(max(p(i, arc), P_FLOOR))

p(i, arc) is the classifier probability for GPS point i and that arc. If the
classifier did not score the arc (the two candidate lists can differ slightly) the
probability is P_FLOOR. weight = 0 adds exactly 0, i.e. HMM only.

No route or reference labels are used.

Configuration (read at call time)
  PATHSNAP_MODEL              saved model bundle (default models/classifier_selected.joblib)
  PATHSNAP_CLASSIFIER_WEIGHT  weight of the classifier log-score (default 1.0)
  PATHSNAP_HMM_CONFIG         optional HMMConfig JSON (default: the HMM's own defaults)

Shared interface:   match(trace, road_network) -> one arc id per GPS point
Experiments:        match_with(trace, road_network, weight, bundle)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.classifier_matcher import candidate_scores, load_bundle
from src.hmm_matcher import HMMConfig, build_matcher

P_FLOOR = 1e-3
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "classifier_selected.joblib"


def model_path() -> Path:
    return Path(os.environ.get("PATHSNAP_MODEL", DEFAULT_MODEL))


def classifier_weight() -> float:
    return float(os.environ.get("PATHSNAP_CLASSIFIER_WEIGHT", "1.0"))


def hmm_config() -> HMMConfig | None:
    path = os.environ.get("PATHSNAP_HMM_CONFIG")
    return HMMConfig.from_json(path) if path else None


def match_with(trace, road_network, weight: float, bundle: dict | None = None,
               hmm_cfg: HMMConfig | None = None, scores=None) -> list[int]:
    """Classifier + HMM with explicit weight, model bundle and HMM settings (experiments).

    scores: optional precomputed (candidates, proba) from candidate_scores, so a
    parameter search does not re-run the classifier for every setting.
    """
    trace = np.atleast_2d(np.asarray(trace, dtype=float))
    nodes = np.asarray(road_network["nodes"], dtype=float)
    arcs = np.atleast_2d(np.asarray(road_network["arcs"], dtype=int))
    network = {"nodes": nodes, "arcs": arcs}

    c, proba = scores if scores is not None else candidate_scores(trace, network, bundle)
    lookup = {(int(p), int(a)): float(q) for p, a, q in zip(c.point, c.arc, proba)}

    def emission_extra_fn(i, arc_ids):
        probs = [max(lookup.get((int(i), int(a)), P_FLOOR), P_FLOOR) for a in arc_ids]
        return weight * np.log(np.asarray(probs, dtype=float))

    trip = {"track": trace, "nodes": nodes, "arcs": arcs}
    result = build_matcher(trip, hmm_cfg).match(trace, emission_extra_fn)
    return result.arc_ids.tolist()


def match(trace, road_network) -> list[int]:
    """match(trace, road_network) -> one arc id per GPS point (-1 = unmatched)."""
    bundle = load_bundle(str(model_path()))
    return match_with(trace, road_network, classifier_weight(), bundle, hmm_config())
