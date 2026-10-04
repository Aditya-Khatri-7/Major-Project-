"""Sanity-check datasets before training or evaluating.

  Image folders (train/valid/test each with real/ and fake/):
      python data/check_datasets.py image --data-dir D:/archive
  Processed text CSVs (class balance, duplicates and question leakage across splits):
      python data/check_datasets.py text --dir data/processed/text
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from eval.datasets import CALIB_PERCENT, IMAGE_EXTENSIONS, image_subset  # noqa: E402


def check_image(args) -> int:
    problems = 0
    for split in ("train", "valid", "test"):
        for cls in ("real", "fake"):
            folder = args.data_dir / split / cls
            if not folder.exists():
                print(f"MISSING  {folder}")
                problems += 1
                continue
            files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
            print(f"{split:>5}/{cls:<4}: {len(files):>8} images")
            sample = random.Random(0).sample(files, min(args.verify, len(files)))
            bad, sizes = 0, []
            for p in sample:
                try:
                    with Image.open(p) as img:
                        img.verify()
                    with Image.open(p) as img:
                        sizes.append(img.size)
                except Exception as exc:
                    bad += 1
                    print(f"  CORRUPT {p}: {exc}")
            problems += bad
            if sizes:
                print(f"           sampled {len(sizes)} files: min side {min(min(s) for s in sizes)}px, max side {max(max(s) for s in sizes)}px")
    valid = args.data_dir / "valid"
    if valid.exists():
        subsets = Counter()
        for p in valid.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                subsets[image_subset(p.relative_to(valid).as_posix())] += 1
        print(f"valid/ split: early-stop={subsets['es']}, calibration={subsets['calib']} (calib target {CALIB_PERCENT}%)")
    print("OK" if not problems else f"{problems} problem(s) found")
    return 1 if problems else 0


def check_text(args) -> int:
    files = sorted(args.dir.glob("*.csv"))
    if not files:
        print(f"No CSV files in {args.dir}")
        return 1
    texts_by_file: dict[str, set[str]] = {}
    groups_by_file: dict[str, set[str]] = {}
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        labels = Counter(int(r["label"]) for r in rows)
        words = sorted(len(r["text"].split()) for r in rows) or [0]
        print(f"{f.name:<24} rows={len(rows):>7} human={labels[0]:>6} ai={labels[1]:>6} "
              f"words median={words[len(words) // 2]} p95={words[int(len(words) * 0.95)]}")
        texts_by_file[f.name] = {r["id"] for r in rows}
        groups_by_file[f.name] = {r["group"] for r in rows if r.get("group")}
    problems = 0
    names = list(texts_by_file)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.startswith("hc3_") and b.startswith("hc3_"):
                dup = len(texts_by_file[a] & texts_by_file[b])
                leak = len(groups_by_file[a] & groups_by_file[b])
                if dup or leak:
                    problems += 1
                    print(f"LEAKAGE {a} vs {b}: {dup} duplicate texts, {leak} shared questions")
    print("OK: no train/val/calib/test leakage among HC3 splits" if not problems else f"{problems} leakage problem(s)")
    return 1 if problems else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--verify", type=int, default=200, help="images to open and verify per folder")
    p.set_defaults(fn=check_image)
    p = sub.add_parser("text")
    p.add_argument("--dir", type=Path, default=Path("data/processed/text"))
    p.set_defaults(fn=check_text)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
