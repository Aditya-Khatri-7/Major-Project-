"""All figures used in training and evaluation. Clean, print-friendly matplotlib plots saved as PNG."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from eval.metrics import binary_metrics  # noqa: E402

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- training curves
_CURVE_SPECS = [
    ("loss", "Loss", "loss"), ("accuracy", "Accuracy", "acc"), ("f1", "F1 (synthetic class)", "f1"),
    ("roc_auc", "ROC AUC", "auc"), ("precision", "Precision", "precision"), ("recall", "Recall", "recall"),
]


def plot_training_curves(history: dict[str, list], out_dir: str | Path, best_epoch: Optional[int] = None,
                         name: str = "model") -> list[Path]:
    """One panel per metric (train vs validation), the best epoch marked, plus the individual PNGs."""
    out_dir = Path(out_dir)
    epochs = history["epoch"]
    saved = []
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    axes = axes.ravel()
    panels = 0
    for key, title, slug in _CURVE_SPECS:
        tr, va = history.get(f"train_{key}"), history.get(f"val_{key}")
        if tr is None and va is None:
            continue
        for ax in (axes[panels], None):
            target = ax
            if target is None:
                single, target = plt.subplots(figsize=(6, 4))
            if tr is not None:
                target.plot(epochs, tr, marker="o", color=PALETTE[0], label="train")
            if va is not None:
                target.plot(epochs, va, marker="s", color=PALETTE[1], label="validation")
            if best_epoch is not None and best_epoch in epochs:
                target.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"best epoch ({best_epoch})")
            target.set_title(f"{title} per epoch"); target.set_xlabel("epoch"); target.set_ylabel(title)
            target.legend()
            if ax is None:
                single.suptitle(name)
                saved.append(_save(single, out_dir / f"curve_{slug}.png"))
        panels += 1
    if "lr" in history and panels < len(axes):
        axes[panels].plot(epochs, history["lr"], marker="o", color=PALETTE[4])
        axes[panels].set_title("Learning rate"); axes[panels].set_xlabel("epoch"); panels += 1
    if "train_loss" in history and "val_loss" in history and panels < len(axes):
        gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
        axes[panels].plot(epochs, gap, marker="o", color=PALETTE[3])
        axes[panels].axhline(0, color="black", linewidth=0.8)
        axes[panels].set_title("Generalisation gap (val loss - train loss)"); axes[panels].set_xlabel("epoch")
        panels += 1
    for ax in axes[panels:]:
        ax.axis("off")
    fig.suptitle(f"{name}: epoch-wise training study", fontsize=14)
    saved.insert(0, _save(fig, out_dir / "training_curves.png"))
    return saved


# ---------------------------------------------------------------- ROC / PR
def plot_roc_multi(curves: dict[str, tuple[Sequence, Sequence]], out_path: str | Path, title: str = "ROC curve") -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, (label, (y, p)) in enumerate(curves.items()):
        if len(set(np.asarray(y).tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, out_path)


def plot_pr_multi(curves: dict[str, tuple[Sequence, Sequence]], out_path: str | Path,
                  title: str = "Precision-recall curve") -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, (label, (y, p)) in enumerate(curves.items()):
        if len(set(np.asarray(y).tolist())) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{label} (AP {auc(rec, prec):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.set_ylim(0, 1.02); ax.legend(loc="lower left")
    return _save(fig, out_path)


# ---------------------------------------------------------------- confusion / calibration / scores
def plot_confusion(y_true, y_pred, out_path: str | Path, class_names=("authentic", "synthetic"),
                   title: str = "Confusion matrix") -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "counts"), (axes[1], norm, ".2%", "row-normalised")):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1], class_names); ax.set_yticks([0, 1], class_names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} ({sub})"); ax.grid(False)
        for (i, j), v in np.ndenumerate(data):
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    color="white" if v > data.max() / 2 else "black", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path)


def plot_calibration(y_true, y_prob, out_path: str | Path, n_bins: int = 10, title: str = "Reliability diagram") -> Path:
    y, p = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.any():
            xs.append(p[mask].mean()); ys.append(y[mask].mean()); counts.append(mask.sum())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax1.plot(xs, ys, marker="o", color=PALETTE[0], label="model")
    ax1.set_ylabel("Observed fraction synthetic"); ax1.set_title(title); ax1.legend()
    ax2.bar(xs, counts, width=1 / n_bins * 0.9, color=PALETTE[0], alpha=0.7)
    ax2.set_xlabel("Predicted P(synthetic)"); ax2.set_ylabel("count")
    return _save(fig, out_path)


def plot_score_distribution(y_true, y_prob, out_path: str | Path, threshold: Optional[float] = None,
                            title: str = "Score distribution") -> Path:
    y, p = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, color=PALETTE[0], label="authentic (label 0)", density=True)
    ax.hist(p[y == 1], bins=bins, alpha=0.65, color=PALETTE[1], label="synthetic (label 1)", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.2f}")
    ax.set_xlabel("P(synthetic)"); ax.set_ylabel("density"); ax.set_title(title); ax.legend()
    return _save(fig, out_path)


def plot_threshold_sweep(y_true, y_prob, out_path: str | Path, title: str = "Metrics versus threshold") -> Path:
    y, p = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    ts = np.linspace(0.02, 0.98, 49)
    rows = [binary_metrics(y, p, t) for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, key in enumerate(("accuracy", "precision", "recall", "f1")):
        ax.plot(ts, [r[key] for r in rows], label=key, color=PALETTE[i])
    ax.set_xlabel("decision threshold"); ax.set_ylabel("metric"); ax.set_title(title); ax.legend()
    return _save(fig, out_path)


# ---------------------------------------------------------------- comparison / summary
def plot_metric_bars(results: dict[str, dict], out_path: str | Path,
                     metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
                     title: str = "Method comparison") -> Path:
    names = list(results)
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names) + 3), 5))
    for i, metric in enumerate(metrics):
        vals = [results[n].get(metric) or 0.0 for n in names]
        bars = ax.bar(np.arange(len(names)) + i * width, vals, width, label=metric, color=PALETTE[i % len(PALETTE)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(names)) + width * (len(metrics) - 1) / 2, names, rotation=25, ha="right")
    ax.set_ylim(0, 1.08); ax.set_title(title); ax.legend(ncol=len(metrics), loc="lower center", fontsize=8)
    return _save(fig, out_path)


def plot_summary_card(metrics: dict, out_path: str | Path, title: str = "Summary") -> Path:
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc",
            "average_precision", "tpr_at_1pct_fpr", "mcc", "ece", "brier"]
    rows = [[k.replace("_", " "), "n/a" if metrics.get(k) is None or metrics.get(k) != metrics.get(k)
             else f"{metrics[k]:.4f}"] for k in keys if k in metrics]
    rows.append(["samples", str(metrics.get("n", ""))])
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["metric", "value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.4)
    ax.set_title(title, fontweight="bold")
    return _save(fig, out_path)


def generate_report_plots(y_true, y_prob, out_dir: str | Path, name: str, threshold: float = 0.5) -> dict:
    """ROC, PR, confusion matrix, calibration, score distribution, threshold sweep and summary card."""
    out_dir = Path(out_dir)
    y, p = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    metrics = binary_metrics(y, p, threshold)
    plot_roc_multi({name: (y, p)}, out_dir / "roc_curve.png", f"ROC - {name}")
    plot_pr_multi({name: (y, p)}, out_dir / "pr_curve.png", f"Precision-recall - {name}")
    plot_confusion(y, (p >= threshold).astype(int), out_dir / "confusion_matrix.png", title=name)
    plot_calibration(y, p, out_dir / "calibration.png", title=f"Reliability - {name}")
    plot_score_distribution(y, p, out_dir / "score_distribution.png", threshold, f"Scores - {name}")
    plot_threshold_sweep(y, p, out_dir / "threshold_sweep.png", f"Threshold sweep - {name}")
    plot_summary_card(metrics, out_dir / "summary_card.png", name)
    return metrics
