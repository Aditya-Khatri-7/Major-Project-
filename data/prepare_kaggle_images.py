"""Arrange the Kaggle image downloads into the folder layout the training / eval scripts expect.

  WildDeepfake subset (fake/<video>/*.png, real/<video>/*.png) -> split BY VIDEO (no frame leakage) into
      data/processed/image/wilddeepfake/{train,valid,test}/{real,fake}/<video>/*.png  (70 / 15 / 15)
  FF++ c32 frames  -> data/processed/image/ffpp/{real,fake}        (Original vs Deepfakes/FaceSwap/NeuralTextures)
  Celeb-DF v2 images (official Test split) -> data/processed/image/celebdf/{real,fake}

Files are hard-linked (no extra disk space), falling back to copy.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

EXT = {".png", ".jpg", ".jpeg"}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def imgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)


def wilddeepfake(raw: Path, out: Path, seed: int) -> None:
    rng = random.Random(seed)
    for cls in ("real", "fake"):
        vids = sorted(d for d in (raw / cls).iterdir() if d.is_dir())
        rng.shuffle(vids)
        n = len(vids)
        cut = {"train": vids[: int(.7 * n)], "valid": vids[int(.7 * n): int(.85 * n)], "test": vids[int(.85 * n):]}
        for split, vs in cut.items():
            count = 0
            for v in vs:
                for p in imgs(v):
                    link(p, out / split / cls / v.name / p.name)
                    count += 1
            print(f"wilddeepfake {split:>5}/{cls:<4}: {len(vs):>4} videos, {count:>6} images")


def sampled(files: list[Path], n: int, rng: random.Random) -> list[Path]:
    return rng.sample(files, min(n, len(files)))


def ffpp(raw: Path, out: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    base = raw / "FF++C32-Frames"
    real = sampled(imgs(base / "Original"), n, rng)
    per = n // 3
    fake = [p for m in ("Deepfakes", "FaceSwap", "NeuralTextures") for p in sampled(imgs(base / m), per, rng)]
    for cls, files in (("real", real), ("fake", fake)):
        for p in files:
            link(p, out / cls / f"{p.parent.name}_{p.name}")
        print(f"ffpp {cls}: {len(files)} images")


def celebdf(raw: Path, out: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    for cls in ("real", "fake"):
        files = sampled(imgs(raw / "Test" / cls), n, rng)
        for p in files:
            link(p, out / cls / p.name)
        print(f"celebdf {cls}: {len(files)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/image"))
    ap.add_argument("--n", type=int, default=3000, help="images per class for the cross-dataset test sets")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    wilddeepfake(a.raw / "wilddeepfake" / "wilddeepfake_subset", a.out / "wilddeepfake", a.seed)
    ffpp(a.raw / "ffpp", a.out / "ffpp", a.n, a.seed)
    celebdf(a.raw / "celebdf" / "Celeb_V2", a.out / "celebdf", a.n, a.seed)
