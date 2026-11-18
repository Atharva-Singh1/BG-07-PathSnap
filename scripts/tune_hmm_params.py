"""Grid-search HMM decoder settings together with the classifier weight.

Tunes sigma_m, beta_m, candidate_radius_m, max_candidates and the weight of the
classifier log-score. The classifier is run once per trip (scores are cached), so
each setting only re-runs the HMM. Weight 0 = HMM only with that setting.

Writes <out-dir>/hmm_params.csv (sorted by point_accuracy_mean, best first) and
<out-dir>/best_hmm_config.json (HMMConfig of the best row, for PATHSNAP_HMM_CONFIG).

Good practice: tune on --data-dir <train> --max-trips N (use --weights 0 there, the
classifier has seen those trips), then confirm on student_test.

Usage
  python scripts/tune_hmm_params.py --data-dir "<path>/student_test" \
      --sigma 8 12 15 20 --beta 15 30 60 --weights 0 0.5 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.classifier_matcher import candidate_scores, load_bundle  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.hmm_matcher import HMMConfig  # noqa: E402
from src.hybrid_matcher import match_with, model_path  # noqa: E402

METRICS = ["point_accuracy_mean", "macro_f1_mean", "route_continuity_mean", "inference_time_s_mean"]
COLUMNS = ["sigma_m", "beta_m", "candidate_radius_m", "max_candidates", "weight"] + METRICS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--sigma", type=float, nargs="+", default=[15.0])
    ap.add_argument("--beta", type=float, nargs="+", default=[30.0])
    ap.add_argument("--radius", type=float, nargs="+", default=[50.0])
    ap.add_argument("--max-candidates", type=int, nargs="+", default=[8])
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--out-dir", type=Path, default=Path("results/hybrid"))
    ap.add_argument("--max-trips", type=int, default=None)
    args = ap.parse_args(argv)

    bundle = load_bundle(str(args.model or model_path()))
    cache: dict = {}

    def scores_for(trace, net):
        key = hash(np.ascontiguousarray(trace).tobytes())
        if key not in cache:
            cache[key] = candidate_scores(trace, net, bundle)
        return cache[key]

    grid = list(itertools.product(args.sigma, args.beta, args.radius, args.max_candidates, args.weights))
    print(f"{len(grid)} settings")
    rows, failed = [], 0
    for n, (sg, bt, rad, mc, w) in enumerate(grid, 1):
        cfg = HMMConfig(sigma_m=sg, beta_m=bt, candidate_radius_m=rad, max_candidates=mc)
        _, summary = evaluate(
            args.data_dir, max_trips=args.max_trips, log=lambda *_: None,
            matcher=lambda trace, net, cfg=cfg, w=w: match_with(
                trace, net, w, bundle, cfg, scores=scores_for(trace, net)))
        failed += summary["n_trips_failed"]
        row = {"sigma_m": sg, "beta_m": bt, "candidate_radius_m": rad, "max_candidates": mc,
               "weight": w, **{k: summary.get(k) for k in METRICS}}
        rows.append(row)
        print(f"[{n}/{len(grid)}] sigma={sg} beta={bt} radius={rad} maxcand={mc} weight={w}: "
              + "  ".join(f"{k.replace('_mean', '')}={row[k]:.4f}" for k in METRICS[:3]))

    rows.sort(key=lambda r: (r["point_accuracy_mean"] is None, -(r["point_accuracy_mean"] or 0)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "hmm_params.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    best = rows[0]
    best_cfg = HMMConfig(sigma_m=best["sigma_m"], beta_m=best["beta_m"],
                         candidate_radius_m=best["candidate_radius_m"],
                         max_candidates=int(best["max_candidates"]))
    (args.out_dir / "best_hmm_config.json").write_text(json.dumps(asdict(best_cfg), indent=2))
    print(f"best: {({k: best[k] for k in COLUMNS[:5]})}  point_accuracy_mean={best['point_accuracy_mean']}")
    print(f"wrote {args.out_dir / 'hmm_params.csv'} and {args.out_dir / 'best_hmm_config.json'}")
    if failed:
        print(f"WARNING: {failed} trip evaluation(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
