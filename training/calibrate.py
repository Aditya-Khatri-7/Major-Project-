"""Fit calibration, fusion weights and decision thresholds on the CALIBRATION split, then freeze them.

Inputs are score caches made by eval/score_dataset.py on held-out calibration data (never test data):
    python training/calibrate.py --modality text  --cache-dir eval/cache/text_hc3_calib
    python training/calibrate.py --modality image --cache-dir eval/cache/image_wd_calib

What is fitted and written to models/calibration.json (merged with any existing content):
  * text_dl / image_dl : temperature T so that P = sigmoid(raw_logit / T) minimises log-loss
  * text_slm           : logistic map P = sigmoid(a * binoculars_score + b)
  * weights            : per-tool fusion weight = max(0.05, 2 * (validation AUROC - 0.5))
  * thresholds         : t_star (F1-optimal), t_lo / t_hi (uncertain band) of the FUSED score
A report (JSON) and reliability diagrams before/after calibration are written next to the results.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration, recalibrate  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from config import settings  # noqa: E402
from eval.cache import load_cache  # noqa: E402
from eval.metrics import binary_metrics, choose_thresholds, expected_calibration_error, nan_to_none  # noqa: E402
from eval.plots import plot_calibration  # noqa: E402
from training.common import write_json  # noqa: E402

MIN_SAMPLES = 100


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature T > 0 minimising the log-loss of sigmoid(logit / T)."""
    def nll(log_t: float) -> float:
        z = logits / np.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, z) - labels * z))     # -[y log p + (1-y) log(1-p)]
    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def fit_logistic(x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    clf = LogisticRegression(C=1e4, max_iter=1000).fit(x.reshape(-1, 1), labels)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def _raw_arrays(cache, tool: str, feature: str):
    ids = [i for i in cache.ids([tool]) if not cache.samples[i]["verdicts"][tool].error
           and feature in cache.samples[i]["verdicts"][tool].raw_features]
    x = np.array([cache.samples[i]["verdicts"][tool].raw_features[feature] for i in ids], dtype=float)
    y = np.array([cache.samples[i]["label"] for i in ids], dtype=int)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(settings.calibration_path))
    ap.add_argument("--report-dir", type=Path, default=Path("eval/results/calibration"))
    ap.add_argument("--plots-dir", type=Path, default=Path("eval/plots/calibration"))
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--target-npv", type=float, default=0.95)
    ap.add_argument("--weight-cap", action="append", default=[], metavar="TOOL=MAX",
                    help="cap a tool's fusion weight, e.g. text_slm=0.1 (a tool that lowers fused AUROC but adds an independent opinion)")
    args = ap.parse_args()

    caps = {k: float(v) for k, v in (item.split("=") for item in args.weight_cap)}
    cache = load_cache(args.cache_dir, apply_calibration=False)
    print(f"Loaded {len(cache.samples)} samples; tools: {cache.tools}")
    cal = copy.deepcopy(load_calibration(str(args.output)))
    report: dict = {"modality": args.modality, "n_samples": len(cache.samples), "tools": {}}

    # ---- 1. per-tool probability calibration ----
    if args.modality == "text":
        for tool in ("text_dl",):
            if tool in cache.tools:
                x, y = _raw_arrays(cache, tool, "raw_logit")
                if len(x) < MIN_SAMPLES:
                    raise SystemExit(f"{tool}: only {len(x)} usable samples (need >= {MIN_SAMPLES}).")
                t = fit_temperature(x, y)
                before = 1 / (1 + np.exp(-x))
                after = 1 / (1 + np.exp(-x / t))
                cal[tool]["temperature"] = t
                report["tools"][tool] = {"temperature": t, "ece_before": expected_calibration_error(y, before),
                                         "ece_after": expected_calibration_error(y, after), "n": len(x)}
                plot_calibration(y, before, args.plots_dir / "text" / f"{tool}_before.png", title=f"{tool} before calibration")
                plot_calibration(y, after, args.plots_dir / "text" / f"{tool}_after.png", title=f"{tool} after (T={t:.2f})")
        if "text_slm" in cache.tools:
            x, y = _raw_arrays(cache, "text_slm", "binoculars_score")
            if len(x) < MIN_SAMPLES:
                raise SystemExit(f"text_slm: only {len(x)} usable samples (need >= {MIN_SAMPLES}).")
            a, b = fit_logistic(x, y)
            if a >= 0:
                print("WARNING: fitted slope is not negative; Binoculars scores should be LOWER for AI text. Check labels.")
            cal["text_slm"].update({"a": a, "b": b, "calibrated": True})
            p = 1 / (1 + np.exp(-(a * x + b)))
            report["tools"]["text_slm"] = {"a": a, "b": b, "ece_after": expected_calibration_error(y, p), "n": len(x),
                                           "auc_raw_score": float(roc_auc_score(y, -x)) if len(set(y)) > 1 else None}
            plot_calibration(y, p, args.plots_dir / "text" / "text_slm_after.png", title="text_slm after calibration")
    else:
        for tool in ("image_dl", "image_clip"):
            if tool not in cache.tools:
                continue
            x, y = _raw_arrays(cache, tool, "raw_logit")
            if len(x) < MIN_SAMPLES:
                raise SystemExit(f"{tool}: only {len(x)} usable samples (need >= {MIN_SAMPLES}).")
            t = fit_temperature(x, y)
            before, after = 1 / (1 + np.exp(-x)), 1 / (1 + np.exp(-x / t))
            cal[tool]["temperature"] = t
            report["tools"][tool] = {"temperature": t, "ece_before": expected_calibration_error(y, before),
                                     "ece_after": expected_calibration_error(y, after), "n": len(x)}
            plot_calibration(y, before, args.plots_dir / "image" / f"{tool}_before.png", title=f"{tool} before calibration")
            plot_calibration(y, after, args.plots_dir / "image" / f"{tool}_after.png", title=f"{tool} after (T={t:.2f})")

    # ---- 2. fusion weights from per-tool AUROC (on recalibrated scores) ----
    recal = {i: {t: recalibrate(v, cal) for t, v in s["verdicts"].items()} for i, s in cache.samples.items()}
    ids = cache.ids()
    y_all = np.array(cache.labels(ids))
    if len(set(y_all.tolist())) < 2:
        raise SystemExit("Calibration data contains a single class.")
    for tool in cache.tools:
        scores = np.array([0.5 if recal[i][tool].error else recal[i][tool].score for i in ids])
        auc = float(roc_auc_score(y_all, scores))
        cal["weights"][tool] = min(max(0.05, 2.0 * (auc - 0.5)), caps.get(tool, float("inf")))
        report["tools"].setdefault(tool, {}).update({"auroc": auc, "weight": cal["weights"][tool],
                                                     "errors": int(sum(recal[i][tool].error for i in ids))})
        print(f"  {tool}: AUROC={auc:.4f} -> weight {cal['weights'][tool]:.3f}")

    # ---- 3. thresholds of the fused score ----
    fused = []
    for i in ids:
        r = fuse(list(recal[i].values()), args.modality, cache.samples[i]["n_words"], cal)
        fused.append(0.5 if r.fused is None else r.fused)
    fused = np.array(fused)
    th = choose_thresholds(y_all, fused, args.target_precision, args.target_npv)
    cal["thresholds"][args.modality] = th
    fm = binary_metrics(y_all, fused, th["t_star"])
    report["fused"] = {"thresholds": th, "auroc": fm["roc_auc"], "f1_at_t_star": fm["f1"],
                       "ece": fm["ece"], "accuracy_at_t_star": fm["accuracy"]}
    plot_calibration(y_all, fused, args.plots_dir / args.modality / "fused_reliability.png", title="Fused score reliability")
    print(f"  fused: AUROC={fm['roc_auc']:.4f}  F1={fm['f1']:.4f}  thresholds={th}")

    cal["calibrated_at"] = datetime.now(timezone.utc).isoformat()
    cal.setdefault("calibrated_modalities", [])
    if args.modality not in cal["calibrated_modalities"]:
        cal["calibrated_modalities"].append(args.modality)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    write_json(nan_to_none(report), args.report_dir / f"{args.modality}_calibration_report.json")
    print(f"Calibration written to {args.output}")


if __name__ == "__main__":
    main()
