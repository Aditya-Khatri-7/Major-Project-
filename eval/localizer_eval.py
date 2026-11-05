# ruff: noqa: E501
"""Evaluate the pixel-level edit localiser on held-out local edits (different base photos), by edit size, with JPEG robustness.

    python eval/localizer_eval.py [--model models/localizer_v1.pt]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from training.train_localizer import MEAN, STD, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/localizer_v1.pt")
ap.add_argument("--split", default="test")
a = ap.parse_args()
root = Path("data/processed/image/local_edits") / a.split
meta = pd.read_csv(root / "meta.csv")
model = build().cuda().eval()
model.load_state_dict(torch.load(a.model, map_location="cuda"))


def prob_map(img: Image.Image) -> np.ndarray:
    x = torch.from_numpy(((np.asarray(img.convert("RGB").resize((512, 512)), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.sigmoid(model(x)["out"].float())[0, 0].cpu().numpy()


def jpeg(img: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


rows = []
for _, r in meta.iterrows():
    mask = np.asarray(Image.open(root / "mask" / f"{r.id}.png")) > 127
    for cond, fn in (("clean", lambda im: im), ("jpeg75", lambda im: jpeg(im, 75))):
        pe, pr = prob_map(fn(Image.open(root / "edited" / f"{r.id}.png"))), prob_map(fn(Image.open(root / "real" / f"{r.id}.png")))
        pred = pe >= 0.5
        rows.append({"area": r.area, "cond": cond, "pix_auc": roc_auc_score(mask.ravel(), pe.ravel()), "iou": (pred & mask).sum() / max(1, (pred | mask).sum()),
                     "max_edit": float(pe.max()), "max_real": float(pr.max()), "p99_edit": float(np.percentile(pe, 99.5)), "p99_real": float(np.percentile(pr, 99.5)),
                     "real_flag": bool((pr >= 0.5).mean() > 0.002)})
df = pd.DataFrame(rows)
for cond in ("clean", "jpeg75"):
    d = df[df.cond == cond]
    print(f"\n== {cond}   (unedited images flagged at all: {d.real_flag.mean():.1%})")
    print(f"{'edit area':>9} {'image AUC (max prob)':>20} {'pixel AUC':>9} {'IoU@0.5':>8} {'edited flagged':>14}")
    for ar in sorted(d.area.unique()):
        e = d[d.area == ar]
        y = np.r_[np.zeros(len(e)), np.ones(len(e))]
        s = np.r_[e.p99_real, e.p99_edit]
        print(f"{ar:9.0%} {roc_auc_score(y, s):20.3f} {e.pix_auc.mean():9.3f} {e.iou.mean():8.3f} {(e.max_edit >= 0.5).mean():14.1%}")
