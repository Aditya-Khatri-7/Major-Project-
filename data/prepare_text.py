"""Build the processed text CSVs used for training, calibration and testing.

Output columns: id, text, label (0 = human, 1 = AI), group, source, generator, domain, attack

  HC3 (train / val / calib / test, split by QUESTION so no question crosses splits):
      python data/prepare_text.py hc3
  MAGE or any other Hugging Face dataset (cross-generator / cross-domain test):
      python data/prepare_text.py hf --dataset yaful/MAGE --split test --text-col text --label-col label \
          --human-value 1 --src-col src --n 3000 --name mage_test
  RAID (cross-generator and adversarial tests) from a downloaded labeled file (csv / parquet / jsonl):
      python data/prepare_text.py raid --file data/raw/raid_train.csv --n 3000

ALWAYS check the printed class counts and sample texts to confirm the label direction is right.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = ["id", "text", "label", "group", "source", "generator", "domain", "attack"]


def _hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _bucket(key: str) -> int:
    """Stable 0-99 bucket from a string (deterministic split, independent of any RNG)."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    n_ai = sum(int(r["label"]) for r in rows)
    print(f"  wrote {path}: {len(rows)} rows ({n_ai} AI / {len(rows) - n_ai} human)")


def show_samples(rows: list[dict], k: int = 2) -> None:
    for label, name in ((0, "HUMAN"), (1, "AI")):
        for r in [r for r in rows if int(r["label"]) == label][:k]:
            print(f"  [{name}] {r['text'][:160].replace(chr(10), ' ')}...")


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    if not n or n >= len(rows):
        return rows
    rng = random.Random(seed)
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    out = pos[: n // 2] + neg[: n // 2]
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------ HC3
def cmd_hc3(args) -> None:
    path = args.hc3_path
    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename="all.jsonl", repo_type="dataset"))
    print(f"Reading {path}")
    splits: dict[str, list[dict]] = {"train": [], "val": [], "calib": [], "test": []}
    seen: set[str] = set()
    cuts = (args.train_pct, args.train_pct + args.val_pct, args.train_pct + args.val_pct + args.calib_pct)
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            b = _bucket(question)
            split = "train" if b < cuts[0] else "val" if b < cuts[1] else "calib" if b < cuts[2] else "test"
            for label, key, generator in ((0, "human_answers", "human"), (1, "chatgpt_answers", "chatgpt")):
                for answer in item.get(key, []) or []:
                    text = str(answer).strip()
                    if len(text.split()) < args.min_words:
                        continue
                    h = _hash(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    splits[split].append({
                        "id": h, "text": text, "label": label, "group": _hash(question),
                        "source": "hc3", "generator": generator, "domain": item.get("source", "unknown"), "attack": "none",
                    })
    for name, rows in splits.items():
        random.Random(args.seed).shuffle(rows)
        write_csv(rows, args.out_dir / f"hc3_{name}.csv")
    show_samples(splits["train"])


# ------------------------------------------------------------------ generic Hugging Face dataset
def cmd_hf(args) -> None:
    from datasets import load_dataset

    print(f"Loading {args.dataset} (config={args.config}, split={args.split})")
    ds = load_dataset(args.dataset, args.config, split=args.split) if args.config else load_dataset(args.dataset, split=args.split)
    rows, seen = [], set()
    for i, item in enumerate(ds):
        text = str(item[args.text_col]).strip()
        if len(text.split()) < args.min_words:
            continue
        h = _hash(text)
        if h in seen:
            continue
        seen.add(h)
        is_human = str(item[args.label_col]) == str(args.human_value)
        src = str(item.get(args.src_col, "")) if args.src_col else ""
        rows.append({"id": h, "text": text, "label": 0 if is_human else 1, "group": h, "source": args.name,
                     "generator": "human" if is_human else src or "machine", "domain": str(item.get(args.domain_col, "")) if args.domain_col else "",
                     "attack": "none"})
    print(f"  {len(rows)} usable rows; class balance: {sum(int(r['label']) for r in rows)} AI")
    show_samples(rows)
    write_csv(stratified(rows, args.n, args.seed), args.out_dir / f"{args.name}.csv")


# ------------------------------------------------------------------ RAID
def cmd_raid(args) -> None:
    import pandas as pd

    p = Path(args.file)
    df = (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_json(p, lines=True) if p.suffix in {".jsonl", ".json"}
          else pd.read_csv(p))
    missing = {"model", "attack", "generation"} - set(df.columns)
    if missing:
        raise SystemExit(f"{p} lacks columns {missing}. Use the labeled RAID train split (test labels are hidden).")
    df = df[df["generation"].astype(str).str.split().str.len() >= args.min_words].copy()
    df["is_human"] = df["model"].astype(str).str.lower() == "human"
    ai_excluded = {m.strip().lower() for m in args.exclude_models.split(",") if m.strip()}

    def to_rows(frame) -> list[dict]:
        return [{
            "id": _hash(str(r.generation)), "text": str(r.generation).strip(), "label": 0 if r.is_human else 1,
            "group": str(getattr(r, "id", "")), "source": "raid",
            "generator": "human" if r.is_human else str(r.model), "domain": str(getattr(r, "domain", "")),
            "attack": str(r.attack),
        } for r in frame.itertuples()]

    def balanced(frame_ai, frame_h, n: int):
        half = n // 2
        if len(frame_ai):
            groups = frame_ai.groupby(["model", "attack"])
            cap = max(1, -(-half // groups.ngroups))
            frame_ai = frame_ai.sample(frac=1, random_state=args.seed).groupby(["model", "attack"], group_keys=False).head(cap).head(half)
        frame_h = frame_h.sample(n=min(half, len(frame_h)), random_state=args.seed)
        return frame_ai, frame_h

    clean = df[df["attack"].astype(str).str.lower() == "none"]
    cross_ai = clean[(~clean["is_human"]) & (~clean["model"].astype(str).str.lower().isin(ai_excluded))]
    ai, h = balanced(cross_ai, clean[clean["is_human"]], args.n)
    rows = to_rows(ai) + to_rows(h)
    random.Random(args.seed).shuffle(rows)
    print("Cross-generator set (clean text, generators other than: " + ", ".join(sorted(ai_excluded)) + ")")
    show_samples(rows)
    write_csv(rows, args.out_dir / "raid_crossgen.csv")

    attacked = df[df["attack"].astype(str).str.lower() != "none"]
    adv_ai = attacked[~attacked["is_human"]]
    adv_h = attacked[attacked["is_human"]]
    if adv_h.empty:
        print("  note: RAID has no attacked human rows here; using clean human text as the negative class.")
        adv_h = clean[clean["is_human"]]
    ai, h = balanced(adv_ai, adv_h, args.n)
    rows = to_rows(ai) + to_rows(h)
    random.Random(args.seed).shuffle(rows)
    print("Adversarial set (attacked machine text)")
    write_csv(rows, args.out_dir / "raid_adv.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=Path, default=Path("data/processed/text"))
    common.add_argument("--min-words", type=int, default=20)
    common.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("hc3", parents=[common])
    p.add_argument("--hc3-path", type=Path, default=None, help="local all.jsonl (default: download from Hugging Face)")
    p.add_argument("--train-pct", type=int, default=70)
    p.add_argument("--val-pct", type=int, default=10)
    p.add_argument("--calib-pct", type=int, default=10)
    p.set_defaults(fn=cmd_hc3)

    p = sub.add_parser("hf", parents=[common])
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--human-value", default="1", help="value of the label column that means HUMAN")
    p.add_argument("--src-col", default=None, help="column naming the generator / source")
    p.add_argument("--domain-col", default=None)
    p.add_argument("--n", type=int, default=3000, help="stratified sample size (0 = all)")
    p.add_argument("--name", required=True, help="output file stem, e.g. mage_test")
    p.set_defaults(fn=cmd_hf)

    p = sub.add_parser("raid", parents=[common])
    p.add_argument("--file", required=True)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--exclude-models", default="chatgpt", help="generators seen in training (excluded from cross-generator set)")
    p.set_defaults(fn=cmd_raid)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
