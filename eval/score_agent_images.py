"""Score images with the REAL image agent (scope guard, face crop, tool selection) and cache each tool's verdict for calibration/testing.

    python eval/score_agent_images.py --set calib      # calibration mix (DFF valid + OpenFake tiny train/val) -> eval/cache/agent_calib
    python eval/score_agent_images.py --set tests      # held-out tests -> eval/cache/agent_test_<name>
The VLM judge is switched off (no key); tools used: dl, clip, modern, edit. Output dirs get --suffix (default _v2) so old caches are never mixed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ["ENABLED_IMAGE_TOOLS"] = "dl,clip,modern,edit"
os.environ["SKIP_FORENSICS"] = "1"
os.environ["GEMINI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval.cache import append_line, done_ids, make_record  # noqa: E402
from eval.clip_face_v3 import TYPES, collect  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402


def score(samples: list[dict], out: Path) -> None:
    from agents.image_agent.agent import run_image_agent

    have = done_ids(out / "image_modern.jsonl")
    for k, s in enumerate(samples):
        if s["id"] in have:
            continue
        for v in run_image_agent({"input_ref": s["path"]})["tool_verdicts"]:
            append_line(out / f"{v.tool_name}.jsonl", make_record(s, v))
        if k % 200 == 0:
            print(out.name, k, "/", len(samples), flush=True)


def rec(path: str, label: int, **meta) -> dict:
    return {"id": hashlib.md5(path.encode()).hexdigest()[:12], "path": path, "label": label, "meta": meta}


def build_sets(which: str, n: int = 150) -> dict[str, list[dict]]:
    """cache-dir name -> samples. Deterministic, so every script sees the same images."""
    rng = np.random.default_rng(5)
    of = pd.read_csv("data/processed/image/openfake_tiny/meta.csv")
    if which == "calib":
        recs = [rec(p, int(y), src="openfake", model=m) for p, y, m, sp in zip(of.path, of.label == "fake", of.model, of.split) if sp != "test"]
        v = load_image_records("data/processed/image/mix_dff/valid", "all", 1600, 7)
        return {"agent_calib": recs + [rec(s["path"], s["label"], src="mix_dff") for s in v]}
    out = {}
    wiki = collect("wiki")["test"]
    real_dff = [str(p) for p in rng.choice(sorted(wiki.values()), n, replace=False)]
    for t in TYPES:
        fk = collect(t)["test"]
        fa = [str(p) for p in rng.choice(sorted(fk.values()), n, replace=False)]
        out[f"agent_test_dff_{t}"] = [rec(p, 0) for p in real_dff] + [rec(p, 1) for p in fa]
    for name in ("celebdf", "ffpp_seen", "ffpp_unseen"):
        r = load_image_records(f"data/processed/image/tests/{name}", "all", 2 * n, 42)
        out[f"agent_test_{name}"] = [rec(s["path"], s["label"]) for s in r]
    te = of[of.split == "test"]
    out["agent_test_openfake"] = [rec(p, int(y == "fake"), model=m) for p, y, m in zip(te.path, te.label, te.model)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["calib", "tests"], required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--suffix", default="_v2")
    a = ap.parse_args()
    for name, samples in build_sets(a.set, a.n).items():
        uniq = list({x["id"]: x for x in samples}.values())          # dedupe: identical ids would double-count
        score(uniq, Path("eval/cache") / (name + a.suffix))


if __name__ == "__main__":
    main()
