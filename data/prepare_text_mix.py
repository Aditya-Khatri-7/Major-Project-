"""Build the diverse text training mix (HC3 + MAGE-train + RAID) and leakage-free held-out test sets.

RAID is split BY SOURCE DOCUMENT (so attacked copies of one text never straddle train/test). Held out of training:
  generators  cohere, cohere-chat      -> raid_unseen_gen  (attack none)
  attacks     paraphrase, homoglyph, synonym -> raid_unseen_attack (generators seen in training)
  raid_test_all: every test-document row (all generators, all attacks)
MAGE rows are drawn from MAGE's own train/validation splits; mage_test.csv (test split) is untouched.
Outputs in data/processed/text: mix_train.csv, mix_val.csv, mix_calib.csv, raid_test_all.csv, raid_unseen_gen.csv,
raid_unseen_attack.csv
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("data/processed/text")
HELD_GEN = {"cohere", "cohere-chat"}
HELD_ATK = {"paraphrase", "homoglyph", "synonym"}
MIN_WORDS = 30
COLS = ["id", "text", "label", "group", "source", "generator", "domain", "attack"]


def h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def bucket(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def balanced(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    a, b = df[df.label == 1], df[df.label == 0]
    k = min(n // 2, len(a), len(b))
    return pd.concat([a.sample(k, random_state=seed), b.sample(k, random_state=seed)])


def mage(split: str, n: int, seed: int, banned: set[str]) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE", split=split).to_pandas()
    ds["text"] = ds["text"].astype(str).str.strip()
    ds = ds[ds.text.str.split().str.len() >= MIN_WORDS]
    ds["id"] = ds.text.map(h)
    ds = ds[~ds.id.isin(banned)].drop_duplicates("id")
    ds["label"] = (ds["label"].astype(str) != "1").astype(int)     # MAGE: 1 = human (verified from samples)
    ds["group"], ds["source"] = ds.id, "mage"
    ds["generator"] = ds["src"].astype(str)
    ds["domain"], ds["attack"] = "", "none"
    return balanced(ds, n, seed)[COLS]


def main(seed: int = 42) -> None:
    mage_test_ids = set(pd.read_csv(OUT / "mage_test.csv").id)
    # ---- HC3 (already split by question)
    hc3 = {s: pd.read_csv(OUT / f"hc3_{s}.csv") for s in ("train", "val", "calib")}
    # ---- MAGE
    m_train = mage("train", 60000, seed, mage_test_ids)
    m_val = mage("validation", 6000, seed, mage_test_ids)
    m_calib = m_val.sample(frac=0.5, random_state=seed)
    m_val = m_val.drop(m_calib.index)
    # ---- RAID by document
    r = pd.read_csv("data/raw/raid_sample2.csv")
    r = r[r.generation.astype(str).str.split().str.len() >= MIN_WORDS].copy()
    r["text"], r["label"] = r.generation.astype(str).str.strip(), (r.model != "human").astype(int)
    r["bk"] = r.source_id.map(bucket)
    r["id"], r["group"], r["source"], r["generator"] = r.text.map(h), r.source_id, "raid", r.model
    keep = ["id", "text", "label", "group", "source", "generator", "domain", "attack"]
    test_docs = r[r.bk >= 0.8]
    r_test_all = test_docs[keep]
    r_unseen_gen = test_docs[test_docs.generator.isin(HELD_GEN) & (test_docs.attack == "none")]
    r_unseen_gen = pd.concat([r_unseen_gen[keep], test_docs[(test_docs.label == 0) & (test_docs.attack == "none")][keep]])
    r_unseen_atk = test_docs[test_docs.attack.isin(HELD_ATK) & ~test_docs.generator.isin(HELD_GEN)][keep]
    train_docs = r[(r.bk < 0.8) & ~r.generator.isin(HELD_GEN) & ~r.attack.isin(HELD_ATK)]
    ai, hu = train_docs[train_docs.label == 1], train_docs[train_docs.label == 0]
    k = min(len(hu), len(ai), 40000)
    r_train = pd.concat([ai.sample(k, random_state=seed), hu.sample(k, random_state=seed)])[keep]
    r_val = r_train.sample(frac=0.08, random_state=seed)
    r_train = r_train.drop(r_val.index)
    r_calib = r_val.sample(frac=0.5, random_state=seed)
    r_val = r_val.drop(r_calib.index)

    train = pd.concat([hc3["train"].sample(24000, random_state=seed), m_train, r_train])
    val = pd.concat([hc3["val"].sample(3000, random_state=seed), m_val, r_val])
    calib = pd.concat([hc3["calib"].sample(3000, random_state=seed), m_calib, r_calib])
    train, val, calib = (d.drop_duplicates("id").sample(frac=1, random_state=seed)[COLS] for d in (train, val, calib))
    bad = set(r_test_all.id) | set(r_unseen_gen.id) | set(r_unseen_atk.id)
    train, val, calib = (d[~d.id.isin(bad)] for d in (train, val, calib))
    val, calib = val[~val.id.isin(train.id)], calib[~calib.id.isin(train.id) & ~calib.id.isin(val.id)]
    for name, d in (("mix_train", train), ("mix_val", val), ("mix_calib", calib), ("raid_test_all", r_test_all),
                    ("raid_unseen_gen", r_unseen_gen), ("raid_unseen_attack", r_unseen_atk)):
        d = d.drop_duplicates("id") if name.startswith("mix") else d
        if name.startswith("raid_test") or name.startswith("raid_unseen"):
            d = d.sample(min(len(d), 3000), random_state=seed)
        d.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:20} rows={len(d):6} AI={int(d.label.sum()):6} human={int((d.label == 0).sum()):6}")
    print("\nclass balance per source (train):")
    print(train.groupby("source").label.agg(["count", "mean"]).round(3).to_string())
    assert not set(train.id) & set(r_test_all.id), "RAID leakage"
    assert not set(train.id) & mage_test_ids, "MAGE test leakage"
    print("leakage checks passed")


if __name__ == "__main__":
    main()
