"""Report figures from committed result files only (no new experiments).

python scripts/make_figures.py   ->  results/figures/*.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(R / "model_selection/model_comparison.csv")))
names = [r["model"].replace("_", "\n") for r in rows]
f = lambda k: [float(r[k]) for r in rows]


def bar(ax, vals, title, err=None):
    ax.bar(names, vals, yerr=err, capsize=3, color="#4C78A8")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(min(vals) * 0.95, max(vals) * 1.03)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.tick_params(labelsize=7)


fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
bar(axs[0], f("cv_candidate_macro_f1_mean"), "CV macro F1 (candidate level, 5-fold GroupKFold)",
    f("cv_candidate_macro_f1_std"))
bar(axs[1], f("cv_top1_accuracy_mean"), "CV Top-1 accuracy (selection metric)", f("cv_top1_accuracy_std"))
fig.tight_layout(); fig.savefig(OUT / "cv_comparison.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
bar(axs[0], f("test_top1_accuracy_mean"), "student_test Top-1 accuracy")
bar(axs[1], f("test_arc_macro_f1_mean"), "student_test macro F1 (arc level)")
bar(axs[2], f("test_log_loss"), "student_test log loss (lower is better)")
fig.tight_layout(); fig.savefig(OUT / "test_comparison.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
x = range(len(rows))
ax.bar([i - 0.2 for i in x], f("cv_train_top1_accuracy_mean"), 0.4, label="train (CV folds)")
ax.bar([i + 0.2 for i in x], f("cv_top1_accuracy_mean"), 0.4, label="validation (CV folds)")
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=7); ax.set_ylim(0.8, 0.96)
ax.set_title("Train vs CV Top-1 (over/underfitting)", fontsize=10); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "train_vs_cv.png", dpi=150); plt.close(fig)

lc = list(csv.DictReader(open(R / "model_selection/mlp_loss_curve.csv")))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot([int(r["epoch"]) for r in lc], [float(r["train_loss"]) for r in lc], label="training loss")
ax2 = ax.twinx()
ax2.plot([int(r["epoch"]) for r in lc], [float(r["val_accuracy"]) for r in lc], color="#E45756",
         label="validation accuracy (early stopping)")
ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("val accuracy")
ax.set_title("MLP training history (only model with epochs)", fontsize=10)
fig.legend(fontsize=7, loc="center right"); fig.tight_layout(); fig.savefig(OUT / "mlp_loss_curve.png", dpi=150)
plt.close(fig)

s = {n: json.loads((R / f"{n}_summary.json").read_text())["summary"] for n in ["baseline", "hmm"]}
fig, ax = plt.subplots(figsize=(6, 3.2))
labels = ["baseline", "best classifier\n(" + max(rows, key=lambda r: float(r["test_top1_accuracy_mean"]))["model"] + ")", "HMM"]
vals = [s["baseline"]["point_accuracy_mean"], max(f("test_top1_accuracy_mean")), s["hmm"]["point_accuracy_mean"]]
ax.bar(labels, vals, color=["#BAB0AC", "#4C78A8", "#54A24B"])
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_title("student_test Top-1: baseline vs classifier vs HMM", fontsize=10); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(OUT / "matchers_comparison.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
