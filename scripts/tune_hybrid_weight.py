"""Choose the classifier weight of the hybrid matcher on a partition (e.g. student_test).

For each weight, runs src.evaluate.evaluate(data_dir, matcher=...) with
src.hybrid_matcher.match_with(..., weight=w) and writes results/hybrid/weights.csv.
Weight 0 is the HMM-only result.

Usage
  python scripts/tune_hybrid_weight.py --data-dir "<path>/student_test"
  (model: env PATHSNAP_MODEL or --model; default models/classifier_selected.joblib)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.classifier_matcher import load_bundle  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.hybrid_matcher import match_with, model_path  # noqa: E402

COLUMNS = ["weight", "point_accuracy_mean", "macro_f1_mean",
           "route_continuity_mean", "inference_time_s_mean"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None,
                    help="model bundle (default: PATHSNAP_MODEL or models/classifier_selected.joblib)")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 0.5, 1, 2, 4])
    ap.add_argument("--out-dir", type=Path, default=Path("results/hybrid"))
    ap.add_argument("--max-trips", type=int, default=None)
    args = ap.parse_args(argv)

    bundle = load_bundle(str(args.model or model_path()))
    rows, failed = [], 0
    for w in args.weights:
        _, summary = evaluate(args.data_dir, max_trips=args.max_trips, log=lambda *_: None,
                              matcher=lambda trace, net, w=w: match_with(trace, net, w, bundle))
        failed += summary["n_trips_failed"]
        rows.append({"weight": w, **{k: summary.get(k) for k in COLUMNS[1:]}})
        print(f"weight={w}: " + "  ".join(f"{k}={summary.get(k)}" for k in COLUMNS[1:])
              + f"  failed_trips={summary['n_trips_failed']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "weights.csv"
    with out.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=COLUMNS)
        w_.writeheader()
        w_.writerows(rows)
    print(f"wrote {out}")
    if failed:
        print(f"WARNING: {failed} trip evaluation(s) failed; see per-trip errors with src.evaluate")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
