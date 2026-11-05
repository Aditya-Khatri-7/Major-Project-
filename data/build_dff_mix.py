"""Build data/processed/image/mix_dff: the existing multi-source face mix plus DeepFakeFace (train/valid splits only).

    python data/build_dff_mix.py

Files are hard-linked (no extra disk). The DeepFakeFace test split (by file-stem hash, see eval/clip_face_v3.py) is never included.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.clip_face_v3 import TYPES, collect  # noqa: E402

SRC = Path("data/processed/image/mix")
DST = Path("data/processed/image/mix_dff")
rng = random.Random(5)
SPLIT = {"train": "train", "valid": "val"}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    plan = {"train": dict(old=8000, real=6000, fake=3000), "valid": dict(old=1500, real=600, fake=200)}
    wiki = collect("wiki")
    fakes = {t: collect(t) for t in TYPES}
    for sp, n in plan.items():
        for cls in ("real", "fake"):
            old = sorted((SRC / sp / cls).iterdir())
            for p in rng.sample(old, min(n["old"], len(old))):
                link(p, DST / sp / cls / f"old_{p.name}")
        for p in rng.sample(sorted(wiki[SPLIT[sp]].values()), min(n["real"], len(wiki[SPLIT[sp]]))):
            link(p, DST / sp / "real" / f"dff_wiki_{p.name}")
        for t in TYPES:
            for p in rng.sample(sorted(fakes[t][SPLIT[sp]].values()), min(n["fake"], len(fakes[t][SPLIT[sp]]))):
                link(p, DST / sp / "fake" / f"dff_{t}_{p.name}")
        print(sp, {c: len(list((DST / sp / c).iterdir())) for c in ("real", "fake")})


if __name__ == "__main__":
    main()
