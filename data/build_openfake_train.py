"""Extract a training set from the full OpenFake train shards (disjoint from the tiny official test split by image sha256).

    python data/build_openfake_train.py   -> data/processed/image/openfake_train/{meta.csv,*.jpg}
Keeps all fakes from every generator (weak ones first), and about as many reals. Images are resized to <= 768 px.
"""
from __future__ import annotations

import glob
import hashlib
import io
import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

OUT = Path("data/processed/image/openfake_train")
OUT.mkdir(parents=True, exist_ok=True)
test_sha = set()
for f in glob.glob("data/raw/openfake/tiny/core/*.parquet"):
    for im in pq.read_table(f, columns=["image"]).column(0).to_pylist():
        test_sha.add(hashlib.sha256(im["bytes"]).hexdigest())
rows, n_f, n_r = [], 0, 0
rng = random.Random(1)
for f in sorted(glob.glob("data/raw/openfake/full/core/*.parquet")):
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=64, columns=["image", "label", "model", "image_sha256"]):
        for im, lab, mod, sha in zip(*(b.column(i).to_pylist() for i in range(4))):
            if sha in test_sha:
                continue
            fake = lab == "fake"
            if not fake and rng.random() > 0.9:
                continue
            try:
                img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
            except Exception:
                continue
            img.thumbnail((768, 768))
            p = OUT / f"{sha[:16]}.jpg"
            img.save(p, quality=95)
            rows.append({"path": str(p), "label": lab, "model": mod})
            n_f += fake
            n_r += not fake
    print(f, n_f, n_r, flush=True)
pd.DataFrame(rows).to_csv(OUT / "meta.csv", index=False)
print(pd.DataFrame(rows).groupby(["label", "model"]).size())
