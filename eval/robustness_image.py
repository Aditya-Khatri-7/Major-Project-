"""Image robustness test: how much do the face detectors lose when images are degraded like social-media re-uploads?

    python eval/robustness_image.py --data data/processed/image/tests/celebdf --name celebdf --n 300

Takes a balanced sample of real/fake images, applies each degradation, scores image_dl and image_clip, and reports AUROC and
accuracy at the frozen fused threshold (models/calibration.json). Temporary files stay inside the project (eval/cache/_robustness_tmp).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402

TMP = Path("eval/cache/_robustness_tmp")


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def _resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    return img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


DEGRADATIONS = {
    "original": lambda im: im,
    "jpeg_q75": lambda im: _jpeg(im, 75),
    "jpeg_q50": lambda im: _jpeg(im, 50),
    "downscale_0.5x": lambda im: _resize_roundtrip(im, 0.5),
    "blur_r1.5": lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    "screenshot_like": lambda im: _jpeg(_resize_roundtrip(im, 0.6), 60),     # downscale + heavy recompression
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=Path("eval/results/robustness"))
    args = ap.parse_args()

    from agents.image_agent.tools import clip_probe, dl_classifier

    cal = load_calibration()
    t_star = cal["thresholds"]["image"]["t_star"]
    samples = load_image_records(args.data, "all", args.n, 42)
    labels = np.array([s["label"] for s in samples])
    TMP.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for name, fn in DEGRADATIONS.items():
            fused, dl_s, clip_s = [], [], []
            for i, s in enumerate(samples):
                with Image.open(s["path"]) as im:
                    out = fn(im.convert("RGB"))
                p = TMP / f"{i}.png"
                out.save(p)
                v_dl = dl_classifier.score_image(str(p), with_gradcam=False)
                v_clip = clip_probe.score_image(str(p))
                dl_s.append(v_dl.score)
                clip_s.append(v_clip.score)
                f = fuse([v_dl, v_clip], "image", 0, cal).fused
                fused.append(0.5 if f is None else f)         # no usable tool = undecided
            fused_a = np.array(fused)
            results[name] = {
                "fused_auroc": float(roc_auc_score(labels, fused_a)),
                "dl_auroc": float(roc_auc_score(labels, dl_s)),
                "clip_auroc": float(roc_auc_score(labels, clip_s)),
                "fused_accuracy": float(((fused_a >= t_star).astype(int) == labels).mean()),
                "fake_recall": float((fused_a[labels == 1] >= t_star).mean()),
                "real_false_alarm": float((fused_a[labels == 0] >= t_star).mean()),
            }
            print(f"{name:16s} fused AUC {results[name]['fused_auroc']:.3f}  acc {results[name]['fused_accuracy']:.3f}  "
                  f"fake recall {results[name]['fake_recall']:.3f}  false alarm {results[name]['real_false_alarm']:.3f}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps({"n": len(samples), "t_star": t_star, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
