"""Per-tool score cache. Each tool is run ONCE per sample and its verdict is appended to a JSONL file.

Layout of a cache directory (one per dataset split):
    <cache_dir>/text_dl.jsonl  text_slm.jsonl  text_llm.jsonl  [text_llm_reflex.jsonl]
Each line: {"id", "label", "n_words", "meta", "verdict": ToolVerdict as dict}

Ablations, fusion and thresholds are then computed offline from these files, with no model or API
calls, and the cache follows the latest calibration via `recalibrate`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agents.calibration import load_calibration, recalibrate
from agents.schemas import ToolVerdict

REFLEX_SUFFIX = "_reflex"


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_line(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def done_ids(path: str | Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path)}


def make_record(sample: dict, verdict: ToolVerdict, n_words: int = 0) -> dict:
    return {"id": sample["id"], "label": int(sample["label"]), "n_words": n_words,
            "meta": sample.get("meta", {}), "verdict": verdict.model_dump()}


class Cache:
    """All cached tool verdicts of one split, joined by sample id."""

    def __init__(self, samples: dict[str, dict], tools: list[str], reflex: dict[str, dict[str, ToolVerdict]]):
        self.samples = samples          # id -> {"label", "n_words", "meta", "verdicts": {tool: ToolVerdict}}
        self.tools = tools              # tool names present in the cache
        self.reflex = reflex            # judge tool -> {id: ToolVerdict}

    def ids(self, tools: Optional[list[str]] = None) -> list[str]:
        """Ids for which every requested tool has a verdict (default: all cached tools)."""
        need = tools or self.tools
        return [i for i, s in self.samples.items() if all(t in s["verdicts"] for t in need)]

    def labels(self, ids: list[str]) -> list[int]:
        return [self.samples[i]["label"] for i in ids]


def load_cache(cache_dir: str | Path, apply_calibration: bool = True, cal: Optional[dict] = None) -> Cache:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}. Run eval/score_dataset.py first.")
    cal = cal or load_calibration()
    samples: dict[str, dict] = {}
    tools: list[str] = []
    reflex: dict[str, dict[str, ToolVerdict]] = {}
    for path in sorted(cache_dir.glob("*.jsonl")):
        stem = path.stem
        is_reflex = stem.endswith(REFLEX_SUFFIX)
        tool = stem[: -len(REFLEX_SUFFIX)] if is_reflex else stem
        for rec in read_jsonl(path):
            verdict = ToolVerdict(**rec["verdict"])
            if apply_calibration:
                verdict = recalibrate(verdict, cal)
            if is_reflex:
                reflex.setdefault(tool, {})[rec["id"]] = verdict
                continue
            entry = samples.setdefault(rec["id"], {"label": rec["label"], "n_words": rec.get("n_words", 0),
                                                   "meta": rec.get("meta", {}), "verdicts": {}})
            entry["verdicts"][tool] = verdict
        if not is_reflex:
            tools.append(tool)
    if not tools:
        raise FileNotFoundError(f"No tool score files (*.jsonl) in {cache_dir}.")
    return Cache(samples, tools, reflex)
