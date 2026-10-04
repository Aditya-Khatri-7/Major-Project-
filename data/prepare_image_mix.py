"""Build the diverse image training mix and a leakage-free FF++ test set.

train/valid (data/processed/image/mix):  WildDeepfake (video split from prepare_kaggle_images.py)
                                       + FF++ C32 frames (split by target-video id: 70/15/15 -> train/valid/test)
                                       + Celeb-DF v2 official Train / Val folders (balanced sample)
Held-out test sets (never trained on): wilddeepfake/test, ffpp_test (FF++ videos with bucket >= 0.85),
                                       celebdf (Celeb-DF official Test folder, already built).
Per source, fakes are sampled to match the number of reals so no dataset is a class shortcut.
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/processed/image")
EXT = {".jpg", ".jpeg", ".png"}
FF_FAKE = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
rng = random.Random(42)


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def imgs(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXT)


def bucket(name: str) -> float:
    vid = name.split("_")[0]
    return int(hashlib.md5(vid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def put(files, dst: Path, prefix: str) -> None:
    for p in files:
        link(p, dst / f"{prefix}_{p.parent.name}_{p.name}")


def main() -> None:
    mix = OUT / "mix"
    # WildDeepfake (hard links to the already split folders)
    for split in ("train", "valid"):
        for cls in ("real", "fake"):
            n = 0
            for p in imgs(OUT / "wilddeepfake" / split / cls):
                link(p, mix / split / cls / f"wdf_{p.parent.name}_{p.name}")
                n += 1
            print(f"wdf {split}/{cls}: {n}")
    # FF++
    base = RAW / "ffpp/FF++C32-Frames"
    real = imgs(base / "Original")
    parts = {"train": lambda b: b < 0.70, "valid": lambda b: 0.70 <= b < 0.85, "test": lambda b: b >= 0.85}
    for split, ok in parts.items():
        r = [p for p in real if ok(bucket(p.name))]
        fakes = [p for m in FF_FAKE for p in imgs(base / m) if ok(bucket(p.name))]
        fakes = rng.sample(fakes, min(len(fakes), len(r) if split != "test" else 3000))
        if split == "test":
            r = rng.sample(r, min(len(r), 3000))
            dst_r, dst_f = OUT / "ffpp_test/real", OUT / "ffpp_test/fake"
        else:
            dst_r, dst_f = mix / split / "real", mix / split / "fake"
        put(r, dst_r, "ff"); put(fakes, dst_f, "ff")
        print(f"ffpp {split}: real {len(r)} fake {len(fakes)}")
    # Celeb-DF official Train / Val
    cdf = RAW / "celebdf/Celeb_V2"
    for src, split, n in (("Train", "train", 10000), ("Val", "valid", 1500)):
        for cls in ("real", "fake"):
            fs = rng.sample(imgs(cdf / src / cls), n)
            put(fs, mix / split / cls, "cdf")
            print(f"celebdf {split}/{cls}: {len(fs)}")
    for s in ("train", "valid"):
        print(s, {c: len(list((mix / s / c).iterdir())) for c in ("real", "fake")})


if __name__ == "__main__":
    main()
