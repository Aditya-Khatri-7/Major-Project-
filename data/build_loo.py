"""Leave-one-out dataset: the current training mix PLUS one extra FF++ manipulation method (train/valid ids only).

    python data/build_loo.py --add Face2Face      -> data/processed/image/mixed_plus_Face2Face  (test on FaceShifter)
    python data/build_loo.py --add FaceShifter    -> data/processed/image/mixed_plus_FaceShifter (test on Face2Face)
    python data/build_loo.py --add Face2Face FaceShifter -> mixed_all (final model; no zero-shot FF++ method left)

Test-id frames are never added. Files are hard links (no extra disk).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_mixed_image import CROPS, OUT, ffpp_split, link  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="+", required=True, choices=["Face2Face", "FaceShifter"])
    args = ap.parse_args()
    name = "mixed_all" if len(args.add) > 1 else f"mixed_plus_{args.add[0]}"
    dst_root, src_root = OUT / name, OUT / "mixed"
    n = 0
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".png"}:
            link(p, dst_root / p.relative_to(src_root)); n += 1
    added = 0
    for method in args.add:
        for p in sorted((CROPS / method).glob("*.jpg")):
            b = ffpp_split(p.stem)
            if b in ("train", "valid"):
                link(p, dst_root / b / "fake" / f"ff_{method}_{p.name}"); added += 1
    print(f"{name}: {n} base files + {added} {args.add} frames")


if __name__ == "__main__":
    main()
