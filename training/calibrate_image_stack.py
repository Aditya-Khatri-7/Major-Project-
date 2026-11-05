"""Fit the learned image-tool combination (logistic regression on raw tool logits) and its decision thresholds on the calibration cache.

    python training/calibrate_image_stack.py --cache-dir eval/cache/agent_calib
Writes cal['stack']['image'] = {coef, present, bias} and cal['thresholds']['image'] into models/calibration.json.
The cache must come from eval/score_agent_images.py (real image-agent verdicts, including raw_logit).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from config import settings  # noqa: E402
from eval.cache import load_cache  # noqa: E402
from eval.metrics import choose_thresholds  # noqa: E402

TOOLS = ("image_dl", "image_clip", "image_modern")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--target-npv", type=float, default=0.95)
    ap.add_argument("--fake-miss", type=float, default=0.01,
                    help="budget: share of calibration fakes allowed to score at or below t_lo (missing a fake costs more than an extra review)")
    a = ap.parse_args()
    cache = load_cache(a.cache_dir, apply_calibration=False)
    ids = list(cache.samples)
    y = np.array([cache.samples[i]["label"] for i in ids])
    X = np.zeros((len(ids), 2 * len(TOOLS)))
    for r, i in enumerate(ids):
        for c, t in enumerate(TOOLS):
            v = cache.samples[i]["verdicts"].get(t)
            if v is not None and not v.error and "raw_logit" in v.raw_features:
                X[r, c], X[r, len(TOOLS) + c] = float(v.raw_features["raw_logit"]), 1.0
    lr = LogisticRegression(C=a.C, max_iter=3000).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    th = choose_thresholds(y, p, a.target_precision, a.target_npv)
    th["t_lo"] = float(min(np.quantile(p[y == 1], a.fake_miss), th["t_star"] - 0.05))
    print(f"n={len(ids)}  AUROC(calib)={roc_auc_score(y, p):.4f}  coef={dict(zip(TOOLS, lr.coef_[0][:3].round(3)))}  thresholds={th}")
    path = Path(settings.calibration_path)
    cal = json.loads(path.read_text(encoding="utf-8"))
    cal.setdefault("stack", {})["image"] = {"coef": {t: float(lr.coef_[0][k]) for k, t in enumerate(TOOLS)},
                                           "present": {t: float(lr.coef_[0][len(TOOLS) + k]) for k, t in enumerate(TOOLS)},
                                           "bias": float(lr.intercept_[0]), "fitted_on": a.cache_dir.name, "n": len(ids)}
    cal["thresholds"]["image"] = th
    cal["calibrated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    load_calibration.__globals__["_cache"].clear()
    print(f"written to {path}")


if __name__ == "__main__":
    main()
