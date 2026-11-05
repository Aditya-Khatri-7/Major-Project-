"""Measure detection and localisation of real local AI edits (Stable Diffusion inpainting) vs their unedited originals.

    python eval/local_edit_eval.py --split test        # scores real+edited with every tool and the heatmap (cache: eval/cache/)
    python eval/local_edit_eval.py --split test --report
Reports, per edit-size bucket: image-level AUROC per tool, localisation (pixel AUROC of the heatmap vs the true mask, and whether the hottest
region overlaps the true mask), and the false-alarm rate of the 'localised region' finding on UNEDITED images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["SKIP_FORENSICS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
root = Path("data/processed/image/local_edits") / a.split
cache = Path(f"eval/cache/local_edit_{a.split}.json")
meta = pd.read_csv(root / "meta.csv")

if not a.report:
    from agents.image_agent import forensics
    from agents.image_agent.tools import clip_probe, dl_classifier, modern_detector
    rows = json.loads(cache.read_text()) if cache.exists() else {}
    for k, r in meta.iterrows():
        for kind in ("real", "edited"):
            key = f"{r.id}_{kind}"
            if key in rows:
                continue
            p = str(root / kind / f"{r.id}.png")
            img = Image.open(p).convert("RGB")
            heat = forensics.detector_heatmap(img)
            mask = np.asarray(Image.open(root / "mask" / f"{r.id}.png").resize((heat.shape[1], heat.shape[0]))) > 127
            regs = forensics.top_regions(heat)
            hit = False
            if regs and kind == "edited":
                b = regs[0]["box"]
                m = np.zeros_like(mask)
                H, W = heat.shape
                m[int(b[1] * H):int(np.ceil(b[3] * H)), int(b[0] * W):int(np.ceil(b[2] * W))] = True
                hit = bool((m & mask).sum() / max(1, (m | mask).sum()) >= 0.1)
            rows[key] = {"id": r.id, "kind": kind, "area": float(r.area),
                         "modern": modern_detector.score_image(p).score, "dl": dl_classifier.score_image(p, with_gradcam=False).score,
                         "clip": clip_probe.score_image(p).score,
                         "hmax": float(heat.max()), "hmed": float(np.median(heat)),
                         "pix_auc": (float(roc_auc_score(mask.ravel(), heat.ravel())) if kind == "edited" and 0 < mask.mean() < 1 else None),
                         "hit": hit, "n_regions": len(regs),
                         "finding": bool(heat.max() >= 0.8 and heat.max() - np.median(heat) >= 0.25 and regs)}
        if k % 20 == 0:
            cache.write_text(json.dumps(rows))
            print(k, "/", len(meta), flush=True)
    cache.write_text(json.dumps(rows))
    sys.exit(0)

df = pd.DataFrame(json.loads(cache.read_text()).values())
real, ed = df[df.kind == "real"], df[df.kind == "edited"]
print(f"pairs: {len(ed)}  | finding fires on UNEDITED: {real.finding.mean():.1%}  on EDITED: {ed.finding.mean():.1%}")
print(f"{'edit area':>9} {'modern AUC':>10} {'dl AUC':>7} {'clip AUC':>8} {'max-heat AUC':>12} {'pixel AUC':>9} "
      f"{'top-region hit':>14} {'finding recall':>14}")
for ar in sorted(ed.area.unique()):
    e = ed[ed.area == ar]
    y = np.r_[np.zeros(len(real)), np.ones(len(e))]
    f = lambda c: roc_auc_score(y, np.r_[real[c], e[c]])  # noqa: E731
    print(f"{ar:9.0%} {f('modern'):10.3f} {f('dl'):7.3f} {f('clip'):8.3f} {f('hmax'):12.3f} {e.pix_auc.mean():9.3f} "
          f"{e.hit.mean():14.1%} {e.finding.mean():14.1%}")
