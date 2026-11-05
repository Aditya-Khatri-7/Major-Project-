"""Robustness: re-score OpenFake test images after a WhatsApp-style degradation (resize to <=1280 px, JPEG q70).

    python eval/score_degraded.py --n 400   -> eval/cache/agent_test_openfake_degraded_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from eval.score_agent_images import build_sets, score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
src = build_sets("tests")["agent_test_openfake"]
rng = np.random.default_rng(3)
pick = [src[i] for i in rng.choice(len(src), min(a.n, len(src)), replace=False)]
out_dir = Path("data/processed/image/degraded_openfake")
out_dir.mkdir(parents=True, exist_ok=True)
samples = []
for s in pick:
    dst = out_dir / f"{s['id']}.jpg"
    if not dst.exists():
        im = Image.open(s["path"]).convert("RGB")
        im.thumbnail((1280, 1280))
        im.save(dst, quality=70)
    samples.append({**s, "path": str(dst)})
score(samples, Path("eval/cache/agent_test_openfake_degraded_v2"))
