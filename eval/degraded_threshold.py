"""Decision threshold for re-uploaded (degraded) images.

    python eval/degraded_threshold.py

1. Degrade a balanced sample of the CALIBRATION split (downscale, screenshot-like) and fit a Youden-J threshold on the fused score of the
   face tools (EfficientNet-B4 + CLIP probe). Test images are never used for the fit.
2. On held-out test sets, compare the standard frozen threshold with the degraded-image threshold under each degradation:
   false-alarm rate on real faces, share of fakes caught, accuracy.

Writes eval/results/degraded_threshold.json. It does not change models/calibration.json; the result is stored there only if you pass --write.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402
from eval.metrics import best_threshold  # noqa: E402
from eval.robustness_image import DEGRADATIONS  # noqa: E402

TMP = Path("eval/cache/_degraded_tmp")
FIT_DEGRADATIONS = ("downscale_0.5x", "screenshot_like")
TESTS = {"celebdf": "data/processed/image/tests/celebdf", "wd_test": "data/processed/image/wilddeepfake/test",
         "ffpp_seen": "data/processed/image/tests/ffpp_seen"}
EVAL_DEGRADATIONS = ("original", "downscale_0.5x", "screenshot_like")


def fused_scores(records, degradation, cal, dl, clip):
    out = []
    TMP.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(records):
        with Image.open(r["path"]) as im:
            img = DEGRADATIONS[degradation](im.convert("RGB"))
        path = TMP / f"{k}.png"
        img.save(path)
        f = fuse([dl.score_image(str(path), with_gradcam=False), clip.score_image(str(path))], "image", 0, cal).fused
        out.append(0.5 if f is None else f)
    return np.array(out)


def rates(y, s, t):
    pred = s >= t
    return {"false_alarm": float(pred[y == 0].mean()), "fake_caught": float(pred[y == 1].mean()), "accuracy": float((pred == y).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-fit", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--write", action="store_true", help="store the degraded threshold in models/calibration.json (thresholds.image_degraded)")
    args = ap.parse_args()
    from agents.image_agent.tools import clip_probe, dl_classifier

    cal = load_calibration()
    t_std = cal["thresholds"]["image"]["t_star"]
    try:
        fit = load_image_records("data/processed/image/mixed/valid", "calib", args.n_fit, 42)
        y_fit = np.array([r["label"] for r in fit])
        scores = np.concatenate([fused_scores(fit, d, cal, dl_classifier, clip_probe) for d in FIT_DEGRADATIONS])
        y_all = np.concatenate([y_fit] * len(FIT_DEGRADATIONS))
        t_deg = best_threshold(y_all, scores, "youden")
        print(f"standard threshold {t_std:.3f}  ->  degraded-image threshold {t_deg:.3f}  "
              f"(fitted on {len(y_all)} degraded calibration images)")
        result = {"threshold_standard": t_std, "threshold_degraded": t_deg, "fit_images": int(len(y_all)),
                  "fit_on": list(FIT_DEGRADATIONS), "tests": {}}
        for name, folder in TESTS.items():
            recs = load_image_records(folder, "all", args.n_test, 42)
            y = np.array([r["label"] for r in recs])
            result["tests"][name] = {}
            for d in EVAL_DEGRADATIONS:
                s = fused_scores(recs, d, cal, dl_classifier, clip_probe)
                result["tests"][name][d] = {"standard": rates(y, s, t_std), "degraded_threshold": rates(y, s, t_deg)}
                a, b = result["tests"][name][d]["standard"], result["tests"][name][d]["degraded_threshold"]
                print(f"{name:10s} {d:15s} false alarms {a['false_alarm']:.2f} -> {b['false_alarm']:.2f}   "
                      f"fakes caught {a['fake_caught']:.2f} -> {b['fake_caught']:.2f}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    out = Path("eval/results/degraded_threshold.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.write:
        cal_path = Path("models/calibration.json")
        c = json.loads(cal_path.read_text(encoding="utf-8"))
        c["thresholds"]["image_degraded"] = {"t_star": t_deg, "note": "fitted on degraded calibration images (eval/degraded_threshold.py)"}
        cal_path.write_text(json.dumps(c, indent=2), encoding="utf-8")
        print("written to models/calibration.json")
    print("saved", out)


if __name__ == "__main__":
    main()
