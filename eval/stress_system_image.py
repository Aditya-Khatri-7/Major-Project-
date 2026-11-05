"""Stress test at SYSTEM level (face crop + CNN + CLIP probe + frozen fusion), not tool level.

    python eval/stress_system_image.py --data data/processed/image/tests/dff --name dff --n 100

For each condition: fused AUROC, share of fakes called authentic, real faces called synthetic, escalation rate, and the
'silent failure' rate = wrong verdict that was NOT escalated to a human (the dangerous case).
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from agents.image_agent.face_crop import crop_face_if_needed  # noqa: E402
from agents.image_agent.tools import clip_probe, dl_classifier  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402
from eval.stress_image import PERTURB, fgsm  # noqa: E402

TMP = Path("eval/cache/_stress_sys_tmp")
CONDITIONS = ["original", "pad_1.8x_context", "rotate_90", "rotate_15", "gauss_noise_s8", "tiny_64px", "watermark_text",
              "grayscale", "fgsm_eps2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()
    cal = load_calibration()
    samples = load_image_records(args.data, "all", args.n, 42)
    y = np.array([s["label"] for s in samples])
    TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    try:
        for cond in CONDITIONS:
            res = []
            for i, s in enumerate(samples):
                p = TMP / f"{i}.png"
                if cond == "fgsm_eps2":
                    fgsm(s["path"], 2 / 255, to_real=s["label"] == 1).save(p)
                else:
                    with Image.open(s["path"]) as im:
                        PERTURB[cond](im.convert("RGB")).save(p)
                path = str(p)
                if not args.no_crop:
                    path = crop_face_if_needed(path) or path
                res.append(fuse([dl_classifier.score_image(path, with_gradcam=False), clip_probe.score_image(path)], "image", 0, cal))
            f = np.array([0.5 if r.fused is None else r.fused for r in res])
            ver = np.array([r.verdict for r in res]); esc = np.array([r.escalate for r in res])
            wrong = ((ver == "synthetic") & (y == 0)) | ((ver == "authentic") & (y == 1))
            out[cond] = {
                "auc": float(roc_auc_score(y, f)),
                "fakes_called_authentic": float((ver[y == 1] == "authentic").mean()),
                "real_called_synthetic": float((ver[y == 0] == "synthetic").mean()),
                "escalated": float(esc.mean()),
                "silent_wrong": float((wrong & ~esc).mean()),
            }
            print(f"{cond:18s}", {k: round(v, 3) for k, v in out[cond].items()}, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    Path("eval/results/stress").mkdir(parents=True, exist_ok=True)
    suffix = "_nocrop" if args.no_crop else ""
    Path(f"eval/results/stress/system_image_{args.name}{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
