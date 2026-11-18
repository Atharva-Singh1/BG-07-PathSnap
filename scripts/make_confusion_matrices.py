"""Confusion matrices (candidate level) on student_test from saved model files.

Each candidate row (GPS point, nearby arc) is a binary case: is this arc the point's
reference arc? Predicted positive = probability >= 0.5. Only the trace and road
network go into the model; reference labels are used only to score.

python scripts/make_confusion_matrices.py --test-dir "<path>/student_test" \
    --model random_forest=results/experiments/random_forest/model.joblib \
    --model mlp=results/experiments/mlp/model.joblib
-> results/figures/confusion_matrices.png, results/confusion_matrices.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import CandidateConfig  # noqa: E402
from src.train_models import build_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--test-dir", required=True)
ap.add_argument("--model", action="append", required=True, help="name=path/to/model.joblib")
a = ap.parse_args()

test = build_dataset(a.test_dir, CandidateConfig())
models = [m.split("=", 1) for m in a.model]
fig, axs = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.2))
axs = [axs] if len(models) == 1 else axs
rows = []
for ax, (name, path) in zip(axs, models):
    bundle = joblib.load(path)
    proba = bundle["model"].predict_proba(test["X"])[:, 1]
    cm = confusion_matrix(test["y"], (proba >= 0.5).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rows.append({"model": name, "model_file": Path(path).as_posix(), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                 "recall_correct_arc": tp / (tp + fn), "precision_correct_arc": tp / (tp + fp)})
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.2f})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1], ["pred: other", "pred: correct"], fontsize=7)
    ax.set_yticks([0, 1], ["true: other", "true: correct"], fontsize=7)
    ax.set_title(name.replace("_", " "), fontsize=9)
fig.suptitle(f"Candidate-level confusion matrices, student_test ({len(test['y']):,} candidate rows, "
             "row-normalised in brackets)", fontsize=9)
fig.tight_layout()
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results/figures/confusion_matrices.png", dpi=150)
with open(ROOT / "results/confusion_matrices.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['model']:<24} tn={r['tn']} fp={r['fp']} fn={r['fn']} tp={r['tp']} "
          f"recall={r['recall_correct_arc']:.3f} precision={r['precision_correct_arc']:.3f}")
