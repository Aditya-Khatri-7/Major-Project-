"""Binary-classification metrics, thresholding, calibration error and bootstrap confidence intervals.

Convention everywhere: label 1 = synthetic (AI text / fake image) = the positive class, and every
score is P(synthetic) in [0, 1].
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def _arrays(y_true: Sequence, y_prob: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob differ in shape: {y.shape} vs {p.shape}")
    return y, p


def expected_calibration_error(y_true: Sequence, y_prob: Sequence, n_bins: int = 10) -> float:
    """ECE of the positive-class probability (equal-width bins)."""
    y, p = _arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def tpr_at_fpr(y_true: Sequence, y_prob: Sequence, target_fpr: float) -> float:
    y, p = _arrays(y_true, y_prob)
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target_fpr, fpr, tpr))


def binary_metrics(y_true: Sequence, y_prob: Sequence, threshold: float = 0.5) -> dict:
    """Full metric set at one decision threshold (AUC / AP / ECE are threshold-free)."""
    y, p = _arrays(y_true, y_prob)
    pred = (p >= threshold).astype(int)
    single_class = len(set(y.tolist())) < 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if not single_class else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if not single_class else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if not single_class else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if not single_class else float("nan"),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, p, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(y, p, 0.05),
        "ece": expected_calibration_error(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positive_rate": float(y.mean()),
    }
    return out


def best_threshold(y_true: Sequence, y_prob: Sequence, criterion: str = "f1") -> float:
    """Threshold maximising F1 (or Youden's J) over the observed scores. Returns 0.5 if undefined."""
    y, p = _arrays(y_true, y_prob)
    if len(set(y.tolist())) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([p, [0.5]]))
    best, best_value = 0.5, -math.inf
    for t in candidates:
        pred = (p >= t).astype(int)
        if criterion == "youden":
            tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
            value = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        else:
            value = f1_score(y, pred, zero_division=0)
        if value > best_value + 1e-12:
            best, best_value = float(t), value
    return best


def choose_thresholds(y_true: Sequence, y_prob: Sequence, target_precision: float = 0.95,
                      target_npv: float = 0.95, min_margin: float = 0.05) -> dict:
    """Decision thresholds from a validation set.

    t_star: F1-optimal cut. t_hi: lowest score whose 'synthetic' calls reach target_precision.
    t_lo: highest score whose 'authentic' calls reach target_npv. Scores between t_lo and t_hi are
    'uncertain'. The band always contains t_star and is at least `min_margin` wide on each side, so
    borderline scores are always flagged even when the validation data separates perfectly.
    """
    y, p = _arrays(y_true, y_prob)
    t_star = best_threshold(y, p, "f1")
    t_hi, t_lo = t_star, t_star
    for t in np.unique(p):
        called = p >= t
        if called.sum() >= 5 and y[called].mean() >= target_precision:
            t_hi = float(t)
            break
    for t in np.unique(p)[::-1]:
        called = p <= t
        if called.sum() >= 5 and (1 - y[called]).mean() >= target_npv:
            t_lo = float(t)
            break
    return {"t_lo": max(0.0, min(t_lo, t_star - min_margin)), "t_hi": min(1.0, max(t_hi, t_star + min_margin)),
            "t_star": t_star}


def bootstrap_ci(y_true: Sequence, y_prob: Sequence, metric: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI of `metric(y, p)`. NaN resamples (single class) are skipped."""
    y, p = _arrays(y_true, y_prob)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        v = metric(y[idx], p[idx])
        if not math.isnan(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))


def selective_metrics(y_true: Sequence, y_pred: Sequence, escalated: Sequence) -> dict:
    """Accuracy on the cases the system decides itself (not escalated) versus coverage."""
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    esc = np.asarray(escalated).astype(bool)
    auto = ~esc
    return {
        "coverage": float(auto.mean()),
        "escalation_rate": float(esc.mean()),
        "accuracy_on_decided": float((y[auto] == pred[auto]).mean()) if auto.any() else float("nan"),
        "accuracy_on_escalated": float((y[esc] == pred[esc]).mean()) if esc.any() else float("nan"),
        "n_decided": int(auto.sum()), "n_escalated": int(esc.sum()),
    }


def metric_value(name: str, y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    """Single named metric (used for bootstrap)."""
    return binary_metrics(y, p, threshold)[name]


def nan_to_none(obj):
    """Recursively replace NaN with None so results serialise to valid JSON."""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def train_metrics_row(prefix: str, y_true: Sequence, y_prob: Sequence, loss: Optional[float] = None,
                      threshold: float = 0.5) -> dict:
    """Metric subset logged per epoch, keyed `<prefix>_<name>`."""
    m = binary_metrics(y_true, y_prob, threshold)
    row = {f"{prefix}_{k}": m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")}
    if loss is not None:
        row[f"{prefix}_loss"] = float(loss)
    return row
