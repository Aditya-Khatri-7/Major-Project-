"""Build the multi-source image training set and the held-out test sets.

Why: the model trained only on WildDeepfake scored AUC 0.52 on FF++ because FF++ frames were full video frames while
WildDeepfake / Celeb-DF are tight face crops (preprocessing mismatch + single-source training). This script
  1. face-crops FF++ frames (MTCNN, 30% margin) so every source looks the same,
  2. mixes WildDeepfake + FF++ (Deepfakes, FaceSwap, NeuralTextures) + Celeb-DF into train / valid,
  3. keeps test sets disjoint by video id (FF++ id buckets, Celeb-DF official Test split),
  4. keeps Face2Face and FaceShifter OUT of training entirely -> an unseen-manipulation test.

Output (hard links, no extra disk):
  data/processed/image/mixed/{train,valid}/{real,fake}/        training mix
  data/processed/image/tests/ffpp_seen/{real,fake}/           FF++ test ids, methods seen in training
  data/processed/image/tests/ffpp_unseen/{real,fake}/         FF++ test ids, Face2Face + FaceShifter (never trained on)
  data/processed/image/tests/celebdf/{real,fake}/             Celeb-DF official Test split
  data/processed/image/mixed/manifest.csv                     provenance of every file
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.prepare_frames import FaceCropper  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW_FFPP = ROOT / "raw/ffpp/FF++C32-Frames"
RAW_CELEB = ROOT / "raw/celebdf/Celeb_V2"
WILD = ROOT / "processed/image/wilddeepfake"
OUT = ROOT / "processed/image"
CROPS = ROOT / "processed/image/ffpp_crops"
SEEN_METHODS = ["Original", "Deepfakes", "FaceSwap", "NeuralTextures"]
UNSEEN_METHODS = ["Face2Face", "FaceShifter"]


def bucket(i: int) -> str:
    """FF++ ids 0-999 -> train / valid / test (video-level split; no id appears in two buckets)."""
    return "train" if i < 720 else ("valid" if i < 860 else "test")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def crop_ffpp(detector: str, methods: list[str], reverse: bool = False, max_id: int = 1000) -> None:
    cropper = FaceCropper(detector)
    for method in methods:
        dst_dir = CROPS / method
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((RAW_FFPP / method).rglob("*.jpg"), reverse=reverse)
        files = [p for p in files if int(p.stem.split("_")[0]) < max_id]
        for p in tqdm(files, desc=f"crop {method}", unit="img"):
            out = dst_dir / p.name
            if out.exists():
                continue
            frame = cv2.imread(str(p))
            face = cropper.crop(frame) if frame is not None else None
            if face is not None:
                cv2.imwrite(str(out), face, [cv2.IMWRITE_JPEG_QUALITY, 95])


def ffpp_split(name: str):
    """Return (bucket, is_real) or None when a fake pair straddles two buckets (would leak identities)."""
    parts = name.split("_")
    ids = [int(x) for x in parts[:-1]]               # drop the trailing 'f<k>'
    buckets = {bucket(i) for i in ids}
    return buckets.pop() if len(buckets) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--celeb-train-per-class", type=int, default=6000)
    ap.add_argument("--celeb-valid-per-class", type=int, default=1200)
    ap.add_argument("--celeb-test-per-class", type=int, default=3000)
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="crop in reverse order (parallel worker meeting the forward one)")
    ap.add_argument("--max-id", type=int, default=1000, help="crop only frames whose first video id is below this")
    ap.add_argument("--crop-only", nargs="*", default=None, help="only crop these FF++ methods, then exit (for parallel workers)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    manifest: list[dict] = []

    if args.crop_only:
        crop_ffpp(args.detector, args.crop_only, args.reverse, args.max_id)
        return
    if not args.skip_crop:
        crop_ffpp(args.detector, SEEN_METHODS + UNSEEN_METHODS)

    def add(src: Path, dst: Path, source: str, split: str, label: str) -> None:
        link(src, dst)
        manifest.append({"path": str(dst.relative_to(OUT)), "source": source, "split": split, "label": label})

    # 1. WildDeepfake: keep its own train / valid
    for split in ("train", "valid"):
        for label in ("real", "fake"):
            for p in (WILD / split / label).rglob("*"):
                if p.is_file():
                    rel = p.relative_to(WILD / split / label).as_posix().replace("/", "_")
                    add(p, OUT / "mixed" / split / label / f"wd_{rel}", "wilddeepfake", split, label)

    # 2. FF++ (face crops, id-bucket split)
    for method in SEEN_METHODS + UNSEEN_METHODS:
        label = "real" if method == "Original" else "fake"
        for p in sorted((CROPS / method).glob("*.jpg")):
            b = ffpp_split(p.stem)
            if b is None:
                continue
            if method in UNSEEN_METHODS:
                if b == "test":
                    add(p, OUT / "tests/ffpp_unseen" / label / f"{method}_{p.name}", f"ffpp-{method}", "test", label)
                continue
            if b == "test":
                add(p, OUT / "tests/ffpp_seen" / label / f"{method}_{p.name}", f"ffpp-{method}", "test", label)
            else:
                add(p, OUT / "mixed" / b / label / f"ff_{method}_{p.name}", f"ffpp-{method}", b, label)
    # unseen test needs reals too: reuse the FF++ test-id originals
    for p in sorted((CROPS / "Original").glob("*.jpg")):
        if ffpp_split(p.stem) == "test":
            add(p, OUT / "tests/ffpp_unseen/real" / f"Original_{p.name}", "ffpp-Original", "test", "real")

    # 3. Celeb-DF: official Train / Val / Test (disjoint by the dataset authors)
    for raw_split, split, per_class in (("Train", "train", args.celeb_train_per_class),
                                        ("Val", "valid", args.celeb_valid_per_class),
                                        ("Test", "test", args.celeb_test_per_class)):
        for label in ("real", "fake"):
            files = sorted((RAW_CELEB / raw_split / label).glob("*.jpg"))
            rng.shuffle(files)
            for p in files[:per_class]:
                dst = (OUT / "tests/celebdf" / label / p.name) if split == "test" else (OUT / "mixed" / split / label / f"cd_{p.name}")
                add(p, dst, "celebdf", split, label)

    with open(OUT / "mixed" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "split", "label"])
        w.writeheader()
        w.writerows(manifest)

    from collections import Counter
    c = Counter((m["split"], m["source"].split("-")[0], m["label"]) for m in manifest)
    for k in sorted(c):
        print(k, c[k])
    unseen = sum(1 for m in manifest if "ffpp_unseen" in m["path"])
    print(f"total {len(manifest)} files | ffpp_unseen files {unseen}")


if __name__ == "__main__":
    main()
