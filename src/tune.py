"""Bayesian hyperparameter tuning for PathSnap Random Forest."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from src import train_models
from src.train_models import CandidateConfig


SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 50, "high": 300},
    "max_depth": {"type": "int", "low": 6, "high": 40},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
    "max_features": {"type": "float", "low": 0.2, "high": 1.0},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--cap-train-points", type=int, default=20000)
    parser.add_argument("--cap-val-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trials < 1 or args.folds < 2:
        parser.error("--trials must be >= 1 and --folds must be >= 2")
    if args.cap_train_points < 1 or args.cap_val_points < 1:
        parser.error("Point caps must be positive")

    out_dir = Path("results/tuning/random_forest")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Student test is deliberately not loaded or evaluated.
    # The --test-dir argument is accepted for CLI compatibility only.
    print("Building training dataset once...", flush=True)
    ds = train_models.build_dataset(
        args.train_dir,
        CandidateConfig(),
        max_median_ref_dist_m=50.0,
    )

    groups = ds["point_trip"]
    all_points = np.arange(len(ds["point_ref"]))
    rng = np.random.default_rng(args.seed)

    # Fixed fold samples, shared across every Optuna trial.
    fold_samples = []
    splitter = GroupKFold(n_splits=args.folds)
    for train_points, val_points in splitter.split(
        all_points, groups=groups
    ):
        train_sample = train_models.sample_points(
            train_points, args.cap_train_points, rng
        )
        val_sample = train_models.sample_points(
            val_points, args.cap_val_points, rng
        )
        fold_samples.append((
            train_models.rows_of_points(ds, train_sample),
            train_models.rows_of_points(ds, val_sample),
        ))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 300
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 6, 40
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 20
            ),
            "max_features": trial.suggest_float(
                "max_features", 0.2, 1.0
            ),
        }

        accuracies = []
        losses = []
        total_fit_time = 0.0

        for train_rows, val_rows in fold_samples:
            model = RandomForestClassifier(
                **params,
                n_jobs=-1,
                random_state=args.seed,
            )

            start = time.perf_counter()
            model.fit(ds["X"][train_rows], ds["y"][train_rows])
            total_fit_time += time.perf_counter() - start

            proba = model.predict_proba(
                ds["X"][val_rows]
            )[:, 1]

            accuracy = train_models.top1_accuracy(
                ds, val_rows, proba
            )
            _, log_loss = train_models.candidate_metrics(
                ds["y"][val_rows], proba
            )

            accuracies.append(accuracy)
            losses.append(log_loss)

        mean_accuracy = float(np.mean(accuracies))
        mean_loss = float(np.mean(losses))

        trial.set_user_attr("mean_log_loss", mean_loss)
        trial.set_user_attr("fit_time_s", total_fit_time)

        print(
            f"Trial {trial.number + 1}/{args.trials}: "
            f"top1={mean_accuracy:.5f}, "
            f"logloss={mean_loss:.5f}, "
            f"fit={total_fit_time:.1f}s, "
            f"params={params}",
            flush=True,
        )

        return mean_accuracy

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials)

    trial_rows = []
    for trial in study.trials:
        trial_rows.append({
            "number": trial.number,
            **{
                key: trial.params.get(key, "")
                for key in SEARCH_SPACE
            },
            "objective": trial.value,
            "mean_log_loss": trial.user_attrs.get(
                "mean_log_loss", ""
            ),
            "fit_time_s": trial.user_attrs.get(
                "fit_time_s", ""
            ),
            "state": trial.state.name,
        })

    with (out_dir / "trials.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(trial_rows[0])
        )
        writer.writeheader()
        writer.writerows(trial_rows)

    summary = {
        "best_params": study.best_params,
        "best_objective": study.best_value,
        "n_trials": len(study.trials),
        "search_space": SEARCH_SPACE,
        "folds": args.folds,
        "cap_train_points": args.cap_train_points,
        "cap_val_points": args.cap_val_points,
        "seed": args.seed,
        "sampler": "TPESampler(seed=42)",
    }

    (out_dir / "best_params.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nBest parameters:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
