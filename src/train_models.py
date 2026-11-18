"""Train, compare and select candidate classifiers for PathSnap.

Models are defined one per file in configs/models/*.json (each with an owner).

Each row is one (GPS point, candidate arc) pair from `src/features.py`; the
target is 1 if the candidate is the point's reference arc. A classifier's
prediction for a point is its highest-scoring candidate (observation-level
Top-1).

Protocol
  1. Build candidates, features and labels for every trip of the train
     partition (labels from src.evaluate.derive_reference_labels).
  2. 5-fold GroupKFold over train trips (no trip in both train and validation).
     Per fold, at most --cap-train-points training points and --cap-val-points
     validation points are sampled (same sample for every model, fixed seed),
     with all candidates of each sampled point.
  3. Metrics per fold: point Top-1 accuracy, candidate-level macro F1
     (threshold 0.5), log loss, fit time, predict time.
  4. Select the model with the highest mean CV Top-1 accuracy (ties: lower
     log loss). The student_test partition is NOT used for selection.
  5. Refit every model on up to --cap-final-points train points and score it
     as a classifier-only matcher on student_test with src.evaluate.
  6. Save the selected model to --model-out.

Usage
  python -m src.train_models --train-dir "<path>/train" --test-dir "<path>/student_test"
  python -m src.train_models ... --quick          # small caps, smoke test
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.evaluate import derive_reference_labels, score_trip, summarise, write_results
from src.features import FEATURE_NAMES, CandidateConfig, build_candidates, label_candidates, top1


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"
MIN_MODELS_FOR_SELECTION = 5   # the rules require at least five compared models


def _estimator_class(path: str):
    """Resolve 'sklearn.<module>.<Class>'. Only scikit-learn classes are allowed."""
    if not path.startswith("sklearn."):
        raise ValueError(f"estimator must be a scikit-learn class, got {path!r}")
    module, _, cls = path.rpartition(".")
    return getattr(importlib.import_module(module), cls)


def load_model_config(path: Path, seed: int) -> tuple[str, tuple]:
    """One model per JSON file in configs/models/. Fields:

    name       model name used on the command line and in results
    owner      team member responsible for this model
    family     algorithm family (for the report table)
    estimator  scikit-learn class, e.g. "sklearn.linear_model.LogisticRegression"
    params     constructor arguments
    scale      true -> StandardScaler before the estimator
    random_state is set to --seed if the estimator accepts it and params don't set it.
    """
    cfg = json.loads(Path(path).read_text())
    for key in ("name", "owner", "family", "estimator", "params"):
        if key not in cfg:
            raise ValueError(f"{path}: missing field {key!r}")
    cls = _estimator_class(cfg["estimator"])
    params = dict(cfg["params"])
    if "random_state" in cls().get_params() and "random_state" not in params:
        params["random_state"] = seed
    scale = bool(cfg.get("scale", False))

    def factory(cls=cls, params=params, scale=scale):
        est = cls(**params)
        return make_pipeline(StandardScaler(), est) if scale else est

    factory()  # fail now on bad parameter names
    conf = (("StandardScaler + " if scale else "") + cls.__name__ + "("
            + ", ".join(f"{k}={v}" for k, v in cfg["params"].items()) + ")")
    return cfg["name"], (cfg["family"], conf, factory, cfg["owner"])


def make_models(seed: int, config_dir: Path = MODEL_CONFIG_DIR) -> dict:
    """name -> (family, configuration text, factory, owner), one per configs/models/*.json."""
    models = {}
    for path in sorted(Path(config_dir).glob("*.json")):
        name, entry = load_model_config(path, seed)
        if name in models:
            raise ValueError(f"duplicate model name {name!r} in {config_dir}")
        models[name] = entry
    if not models:
        raise FileNotFoundError(f"no model configs in {config_dir}")
    return models


def apply_params(models: dict, params: dict) -> dict:
    """Override hyperparameters: {"logistic_regression": {"C": 0.1}, ...}.

    Parameters are set on the final estimator (after any StandardScaler).
    Unknown model names or parameter names raise an error before training.
    """
    out = dict(models)
    for name, p in params.items():
        if name not in out:
            raise ValueError(f"--params: unknown model {name!r}; choose from {sorted(out)}")
        family, conf, factory, owner = out[name]

        def make(factory=factory, p=p):
            m = factory()
            (m[-1] if hasattr(m, "steps") else m).set_params(**p)
            return m

        make()  # validate parameter names now, not after the dataset is built
        out[name] = (family, f"{conf} | overrides {json.dumps(p)}", make, owner)
    return out


def _experiment_tag(models: list[str] | None, params: dict) -> str:
    parts = list(models or ["all"])
    for name, p in params.items():
        parts += [f"{k}{v}" for k, v in p.items()]
    tag = "_".join(parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in tag)[:120]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def _trip_dict(tid, t):
    return {"trip_id": tid, "track": t["trace"], "nodes": t["road_network"]["nodes"],
            "arcs": t["road_network"]["arcs"], "route": t["route"]}


def build_dataset(data_dir, cfg: CandidateConfig, max_median_ref_dist_m: float | None = None,
                  log=print) -> dict:
    """Stack candidates of every trip; point ids are global across trips.

    If max_median_ref_dist_m is set, trips whose GPS points lie (median) further
    than that from their reference route are excluded: their derived labels are
    unreliable (the route does not match the trace). Used for train only.
    """
    data = load_data(data_dir)
    X, y, row_point, row_arc, point_ref, point_trip = [], [], [], [], [], []
    excluded = []
    offset = 0
    t0 = time.perf_counter()
    for tid, t in data.items():
        ref, dist = derive_reference_labels(_trip_dict(tid, t))
        med = float(np.median(dist))
        if max_median_ref_dist_m is not None and med > max_median_ref_dist_m:
            excluded.append({"trip_id": tid, "points": len(ref), "median_ref_dist_m": round(med, 1)})
            log(f"  excluded {tid}: median distance to reference route {med:.1f} m")
            continue
        gi = len(point_ref)
        c = build_candidates(t["trace"], t["road_network"], cfg)
        X.append(c.X); y.append(label_candidates(c, ref))
        row_point.append(c.point + offset); row_arc.append(c.arc)
        point_ref.append(ref); point_trip.append(np.full(c.n_points, gi))
        offset += c.n_points
    kept = [tid for tid in data if tid not in {e["trip_id"] for e in excluded}]
    ds = {"trip_ids": kept, "excluded_trips": excluded, "X": np.vstack(X), "y": np.concatenate(y),
          "row_point": np.concatenate(row_point), "row_arc": np.concatenate(row_arc),
          "point_ref": np.concatenate(point_ref), "point_trip": np.concatenate(point_trip)}
    hit = np.bincount(ds["row_point"], weights=ds["y"], minlength=offset) > 0
    ds["candidate_miss_rate"] = float(np.mean(~hit))
    log(f"  {Path(data_dir).name}: {len(kept)} trips, {offset} points, {len(ds['y'])} candidate rows, "
        f"candidate miss rate {ds['candidate_miss_rate']:.4f} ({time.perf_counter() - t0:.1f}s)")
    return ds


def rows_of_points(ds, points: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(ds["point_ref"]), dtype=bool)
    mask[points] = True
    return np.flatnonzero(mask[ds["row_point"]])


def sample_points(points: np.ndarray, cap: int | None, rng) -> np.ndarray:
    if cap is None or len(points) <= cap:
        return points
    return np.sort(rng.choice(points, size=cap, replace=False))


def top1_accuracy(ds, rows: np.ndarray, score: np.ndarray) -> float:
    pts = ds["row_point"][rows]
    uniq, local = np.unique(pts, return_inverse=True)
    pred = top1(local, ds["row_arc"][rows], score, len(uniq))
    return float(np.mean(pred == ds["point_ref"][uniq]))


def candidate_metrics(y_true, proba) -> tuple[float, float]:
    y_pred = (proba >= 0.5).astype(int)
    return (float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])))


# --------------------------------------------------------------------------- #
# Main protocol
# --------------------------------------------------------------------------- #

def run(args) -> int:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed
    cfg = CandidateConfig()
    models = apply_params(make_models(args.seed), args.params)
    if args.models:
        unknown = set(args.models) - set(models)
        if unknown:
            raise ValueError(f"--models: unknown {sorted(unknown)}; choose from {sorted(models)}")
        models = {k: v for k, v in models.items() if k in args.models}

    print("Building train dataset ...")
    train = build_dataset(args.train_dir, cfg, max_median_ref_dist_m=args.max_median_ref_dist)
    print("Building student_test dataset ...")
    test = build_dataset(args.test_dir, cfg)

    # ---------------- cross-validation on train ----------------
    folds = list(GroupKFold(n_splits=args.folds).split(
        np.zeros(len(train["point_ref"])), groups=train["point_trip"]))
    fold_samples = []
    rng = np.random.default_rng(rng_seed)
    for tr_pts, va_pts in folds:
        tr = sample_points(tr_pts, args.cap_train_points, rng)
        va = sample_points(va_pts, args.cap_val_points, rng)
        fold_samples.append((rows_of_points(train, tr), rows_of_points(train, va), len(tr), len(va)))

    cv_rows = []
    for name, (family, conf, factory, owner) in models.items():
        for f, (tr_rows, va_rows, n_tr, n_va) in enumerate(fold_samples):
            model = factory()
            t0 = time.perf_counter()
            model.fit(train["X"][tr_rows], train["y"][tr_rows])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            proba = model.predict_proba(train["X"][va_rows])[:, 1]
            pred_s = time.perf_counter() - t0
            f1m, ll = candidate_metrics(train["y"][va_rows], proba)
            acc = top1_accuracy(train, va_rows, proba)
            cv_rows.append({"model": name, "fold": f, "train_points": n_tr, "val_points": n_va,
                            "top1_accuracy": acc, "candidate_macro_f1": f1m, "log_loss": ll,
                            "fit_time_s": fit_s, "predict_time_s": pred_s})
            print(f"  CV {name:<24} fold {f}: top1={acc:.4f} candF1={f1m:.4f} logloss={ll:.4f} "
                  f"fit={fit_s:.1f}s")
    _write_csv(out / "cv_folds.csv", cv_rows)

    cv_summary = []
    for name, (family, conf, _, owner) in models.items():
        r = [x for x in cv_rows if x["model"] == name]
        row = {"model": name, "owner": owner, "family": family, "configuration": conf}
        for k in ["top1_accuracy", "candidate_macro_f1", "log_loss", "fit_time_s", "predict_time_s"]:
            v = np.array([x[k] for x in r])
            row[f"cv_{k}_mean"], row[f"cv_{k}_std"] = float(v.mean()), float(v.std())
        cv_summary.append(row)
    selected = sorted(cv_summary, key=lambda r: (-r["cv_top1_accuracy_mean"],
                                                 r["cv_log_loss_mean"]))[0]["model"]
    print(f"Selected by mean CV Top-1 accuracy: {selected}")

    # ---------------- refit and score on student_test ----------------
    final_pts = sample_points(np.arange(len(train["point_ref"])), args.cap_final_points,
                              np.random.default_rng(rng_seed + 1))
    final_rows = rows_of_points(train, final_pts)
    test_data = load_data(args.test_dir)
    test_summary = []
    for name, (family, conf, factory, owner) in models.items():
        model = factory()
        t0 = time.perf_counter()
        model.fit(train["X"][final_rows], train["y"][final_rows])
        fit_s = time.perf_counter() - t0
        proba_all = model.predict_proba(test["X"])[:, 1]
        f1m, ll = candidate_metrics(test["y"], proba_all)

        trip_rows = []
        for tid, t in test_data.items():
            trip = _trip_dict(tid, t)
            ref, dist = derive_reference_labels(trip)
            t0 = time.perf_counter()
            c = build_candidates(t["trace"], t["road_network"], cfg)
            pred = top1(c.point, c.arc, model.predict_proba(c.X)[:, 1], c.n_points)
            inf_s = time.perf_counter() - t0
            trip_rows.append(score_trip(trip, pred, ref, dist, inf_s))
        summ = summarise(trip_rows)
        write_results(trip_rows, summ, out, f"test_{name}",
                      {"run_name": f"test_{name}", "source": f"classifier-only:{name}",
                       "partition": Path(args.test_dir).name,
                       "trip_ids": [r["trip_id"] for r in trip_rows]})
        cvr = next(r for r in cv_summary if r["model"] == name)
        row = {**cvr, "final_train_points": len(final_pts), "final_fit_time_s": fit_s,
               "test_top1_accuracy_mean": summ["point_accuracy_mean"],
               "test_top1_accuracy_pooled": summ["point_accuracy_pooled"],
               "test_arc_macro_f1_mean": summ["macro_f1_mean"],
               "test_candidate_macro_f1": f1m, "test_log_loss": ll,
               "test_route_continuity_mean": summ["route_continuity_mean"],
               "test_inference_time_s_per_trip": summ["inference_time_s_mean"]}
        test_summary.append(row)
        print(f"  TEST {name:<24} top1={row['test_top1_accuracy_mean']:.4f} "
              f"arcF1={row['test_arc_macro_f1_mean']:.4f} logloss={ll:.4f} "
              f"cont={row['test_route_continuity_mean']:.4f}")

        est = model[-1] if hasattr(model, "steps") else model
        if getattr(est, "loss_curve_", None) is not None:   # real epoch history (e.g. MLP)
            val = getattr(est, "validation_scores_", None)
            _write_csv(out / f"{name}_loss_curve.csv",
                       [{"epoch": i + 1, "train_loss": l,
                         "val_accuracy": val[i] if val is not None else ""}
                        for i, l in enumerate(est.loss_curve_)])
        if name == selected:
            Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "model_name": name, "feature_names": FEATURE_NAMES,
                         "candidate_config": cfg.to_dict(), "seed": args.seed},
                        args.model_out, compress=3)

    _write_csv(out / "model_comparison.csv", test_summary)
    meta = {"selected_model": selected,
            "selection_rule": "highest mean 5-fold GroupKFold CV Top-1 accuracy on train "
                              "(ties: lower CV log loss); student_test not used for selection",
            "models_run": list(models), "param_overrides": args.params,
            "seed": args.seed, "folds": args.folds,
            "cap_train_points": args.cap_train_points, "cap_val_points": args.cap_val_points,
            "cap_final_points": args.cap_final_points,
            "candidate_config": cfg.to_dict(), "features": FEATURE_NAMES,
            "train_trips_used": len(train["trip_ids"]),
            "train_trips_excluded": train["excluded_trips"],
            "exclusion_rule": f"train trips with median GPS-to-reference-route distance > "
                              f"{args.max_median_ref_dist} m (derived labels unreliable)",
            "train_candidate_miss_rate": train["candidate_miss_rate"],
            "test_candidate_miss_rate": test["candidate_miss_rate"],
            "train_points": int(len(train["point_ref"])),
            "test_points": int(len(test["point_ref"])),
            "model_file": Path(args.model_out).as_posix(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "selection.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and compare PathSnap candidate classifiers.")
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/model_selection for a full run of all models with "
                         "default settings; otherwise results/experiments/<tag>")
    ap.add_argument("--model-out", type=Path, default=None,
                    help="default: models/classifier_selected.joblib for a full default run; "
                         "otherwise <out-dir>/model.joblib")
    ap.add_argument("--models", nargs="*", help="subset of model names to run")
    ap.add_argument("--params", type=json.loads, default={},
                    help='hyperparameter overrides as JSON, e.g. {"logistic_regression": {"C": 0.1}}')
    ap.add_argument("--tag", default=None, help="experiment folder name (default: from models/params)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cap-train-points", type=int, default=40_000)
    ap.add_argument("--cap-val-points", type=int, default=20_000)
    ap.add_argument("--cap-final-points", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-median-ref-dist", type=float, default=50.0,
                    help="exclude train trips whose median GPS-to-route distance exceeds this (m)")
    ap.add_argument("--quick", action="store_true", help="tiny caps for a smoke test")
    args = ap.parse_args(argv)
    if args.quick:
        args.cap_train_points, args.cap_val_points, args.cap_final_points = 3000, 1500, 5000

    # Only a full, default run may write the committed selection results and model.
    # Subsets, overrides and quick runs go to results/experiments/<tag> instead.
    default_caps = (args.cap_train_points, args.cap_val_points, args.cap_final_points,
                    args.folds, args.seed) == (40_000, 20_000, 80_000, 5, 42)
    n_configs = len(list(MODEL_CONFIG_DIR.glob("*.json")))
    official = (not args.models and not args.params and default_caps and not args.quick
                and n_configs >= MIN_MODELS_FOR_SELECTION)
    if not official and not args.models and n_configs < MIN_MODELS_FOR_SELECTION:
        print(f"Only {n_configs} model configs in {MODEL_CONFIG_DIR.name}/ "
              f"(need {MIN_MODELS_FOR_SELECTION} for the official selection run).")
    if args.out_dir is None:
        args.out_dir = (Path("results/model_selection") if official else
                        Path("results/experiments") / (args.tag or _experiment_tag(args.models, args.params)
                                                       + ("_quick" if args.quick else "")))
    if args.model_out is None:
        args.model_out = (Path("models/classifier_selected.joblib") if official
                          else Path(args.out_dir) / "model.joblib")
    print(f"Output folder: {args.out_dir}   model file: {args.model_out}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
