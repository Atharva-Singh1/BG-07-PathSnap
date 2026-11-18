"""Classifier-only matcher: the selected candidate classifier, no HMM.

match(trace, road_network) -> one arc id per GPS point: for every point, the
candidate arc with the highest classifier probability.

The model is produced by `python -m src.train_models` and loaded from
MODEL_PATH (default models/classifier_selected.joblib).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib

from src.features import CandidateConfig, build_candidates, top1

MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL",
                                 Path(__file__).resolve().parents[1] / "models"
                                 / "classifier_selected.joblib"))


@lru_cache(maxsize=4)
def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def candidate_scores(trace, road_network, bundle: dict | None = None):
    """Candidates and the classifier's probability for each (for HMM emissions)."""
    bundle = bundle or load_bundle()
    c = build_candidates(trace, road_network, CandidateConfig(**bundle["candidate_config"]))
    return c, bundle["model"].predict_proba(c.X)[:, 1]


def match(trace, road_network) -> list[int]:
    c, proba = candidate_scores(trace, road_network)
    return top1(c.point, c.arc, proba, c.n_points).tolist()
