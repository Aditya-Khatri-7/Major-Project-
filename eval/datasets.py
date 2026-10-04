"""Loaders for the processed evaluation / calibration data (text CSVs, image folders or CSV manifests)."""
from __future__ import annotations

import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Optional

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CALIB_PERCENT = 40      # share of the image `valid/` folder reserved for calibration (not used for early stopping)


def image_subset(rel_path: str) -> str:
    """Deterministic split of a validation image: 'calib' (CALIB_PERCENT %) or 'es' (early stopping)."""
    bucket = int(hashlib.md5(rel_path.replace("\\", "/").encode("utf-8")).hexdigest(), 16) % 100
    return "calib" if bucket < CALIB_PERCENT else "es"


def _stratified_limit(rows: list[dict], limit: Optional[int], seed: int) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for r in rows:
        by_label[r["label"]].append(r)
    for group in by_label.values():
        rng.shuffle(group)
    per_class = limit // 2
    chosen = by_label[0][:per_class] + by_label[1][:per_class]
    if len(chosen) < limit:                                   # one class ran out: top up from the other
        rest = by_label[0][per_class:] + by_label[1][per_class:]
        rng.shuffle(rest)
        chosen += rest[:limit - len(chosen)]
    rng.shuffle(chosen)
    return chosen


def load_text_rows(csv_path: str | Path, limit: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Rows from a processed text CSV: id, text, label (0 human / 1 AI) plus any metadata columns."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = row.get("text") or ""
            if not text.strip():
                continue
            meta = {k: v for k, v in row.items() if k not in {"id", "text", "label"} and v not in (None, "")}
            rows.append({"id": row.get("id") or f"row{i}", "text": text, "label": int(row["label"]), "meta": meta})
    return _stratified_limit(rows, limit, seed)


def load_image_records(source: str | Path, subset: str = "all", limit: Optional[int] = None,
                       seed: int = 42) -> list[dict]:
    """Image records from a folder with real/ and fake/ subfolders, or a CSV manifest (path,label).

    `subset` ('all' | 'es' | 'calib') applies the deterministic validation split; use 'calib' for
    calibration and 'es' for early stopping so the two never overlap.
    """
    source = Path(source)
    records: list[dict] = []
    if source.is_file():
        with open(source, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                records.append({"id": row["path"], "path": row["path"], "label": int(row["label"]), "meta": {}})
    else:
        for name, label in (("real", 0), ("fake", 1)):
            folder = source / name
            if not folder.exists():
                raise FileNotFoundError(f"Expected folder not found: {folder}")
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    rel = p.relative_to(source).as_posix()
                    records.append({"id": rel, "path": str(p), "label": label, "meta": {}})
    if subset != "all":
        records = [r for r in records if image_subset(r["id"]) == subset]
    return _stratified_limit(records, limit, seed)
