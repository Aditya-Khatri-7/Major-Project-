"""Small, quota-aware judge evaluation (for a free API key limited to ~20 requests per day).

    python eval/judge_minirun.py --n-text 8 --n-image 8          # real calls, stops at the first quota error
    python eval/judge_minirun.py --dry-run                        # no API calls: checks the plumbing with fake judge scores

Uses samples that already have local-tool scores in the benchmark caches, so each one can be compared three ways:
local tools only, judge only, and local tools + judge fused with the frozen calibration. It does NOT run Reflexion (extra calls).
The sample is far too small for calibration or confidence intervals; the output says so.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.calibration import load_calibration  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from agents.schemas import ToolVerdict  # noqa: E402
from config import settings  # noqa: E402
from eval.cache import load_cache  # noqa: E402
from eval.datasets import load_image_records, load_text_rows  # noqa: E402

TEXT_SET = ("mage_test", Path("data/processed/text/mage_test.csv"), Path("eval/cache/text_mage_test"))
IMAGE_SET = ("ffpp_unseen", Path("data/processed/image/tests/ffpp_unseen"), Path("eval/cache/image_ffpp_unseen"))
OUT = Path("eval/results/judge_minirun")


def pick(ids_with_labels: list[tuple[str, int]], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by = {0: [], 1: []}
    for i, y in ids_with_labels:
        by[y].append(i)
    for v in by.values():
        rng.shuffle(v)
    half = n // 2
    return by[0][:half] + by[1][:n - half]


def evaluate(modality: str, n: int, dry: bool, seed: int, cal: dict) -> dict:
    name, data, cache_dir = TEXT_SET if modality == "text" else IMAGE_SET
    cache = load_cache(cache_dir, apply_calibration=True)
    local_tools = [t for t in cache.tools if not t.endswith(("_llm", "_vlm"))]
    usable_ids = [i for i in cache.ids(local_tools) if all(not cache.samples[i]["verdicts"][t].error for t in local_tools)]
    ids = pick([(i, cache.samples[i]["label"]) for i in usable_ids], n, seed)
    if modality == "text":
        rows = {r["id"]: r for r in load_text_rows(data)}
        payload = {i: rows[i]["text"] for i in ids if i in rows}
    else:
        recs = {r["id"]: r for r in load_image_records(data)}
        payload = {i: recs[i]["path"] for i in ids if i in recs}
    judge_name = "text_llm" if modality == "text" else "image_vlm"
    if not dry:
        from agents.gemini_client import GeminiQuotaExceeded  # noqa: F401
        from agents.image_agent.tools import vlm_judge
        from agents.text_agent.tools import llm_judge
        score = llm_judge.score_text if modality == "text" else vlm_judge.score_image
    rng = random.Random(seed + 1)
    samples, stopped = [], None
    for i in ids:
        if i not in payload:
            continue
        label = cache.samples[i]["label"]
        local = [cache.samples[i]["verdicts"][t] for t in local_tools]
        if dry:
            p = min(1.0, max(0.0, (0.8 if label else 0.2) + rng.uniform(-0.3, 0.3)))
            judge = ToolVerdict(tool_name=judge_name, modality=modality, score=p, confidence=abs(p - 0.5) * 2, explanation="dry run")
        else:
            judge = score(payload[i])
        if judge.error:
            stopped = judge.explanation[:160]
            break
        n_words = len(payload[i].split()) if modality == "text" else 0
        base = fuse(local, modality, n_words, cal)
        both = fuse(local + [judge], modality, n_words, cal)
        samples.append({"id": i, "label": label, "judge_score": round(judge.score, 3), "judge_confidence": round(judge.confidence, 3),
                        "judge_reason": judge.explanation[:200],
                        "local_fused": None if base.fused is None else round(base.fused, 3), "local_verdict": base.verdict,
                        "with_judge_fused": None if both.fused is None else round(both.fused, 3), "with_judge_verdict": both.verdict,
                        "with_judge_referred": both.escalate, "reflexion_would_trigger": both.needs_reflexion})
    return {"modality": modality, "set": name, "requested": n, "scored": len(samples), "stopped_early": stopped, "samples": samples}


def summarise(res: dict) -> dict:
    s = res["samples"]
    if not s:
        return {"n": 0}
    def correct(key, thr=0.5):
        return sum((x[key] >= thr) == bool(x["label"]) for x in s if x[key] is not None)

    def decided(v):
        return [x for x in s if x[v] != "uncertain"]

    def acc(v):
        d = decided(v)
        return sum((x[v] == "synthetic") == bool(x["label"]) for x in d) / len(d) if d else None
    return {"n": len(s), "judge_accuracy_at_0.5": correct("judge_score") / len(s),
            "local_fused_accuracy_at_0.5": correct("local_fused") / len(s),
            "with_judge_accuracy_at_0.5": correct("with_judge_fused") / len(s),
            "local_decided_accuracy": acc("local_verdict"), "local_decided_n": len(decided("local_verdict")),
            "with_judge_decided_accuracy": acc("with_judge_verdict"), "with_judge_decided_n": len(decided("with_judge_verdict")),
            "reflexion_would_trigger": sum(x["reflexion_would_trigger"] for x in s),
            "judge_mean_score_real": sum(x["judge_score"] for x in s if not x["label"]) / max(1, sum(1 for x in s if not x["label"])),
            "judge_mean_score_fake": sum(x["judge_score"] for x in s if x["label"]) / max(1, sum(1 for x in s if x["label"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-text", type=int, default=8)
    ap.add_argument("--n-image", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not settings.llm_configured:
        print("No judge key configured."); return 2
    cal = load_calibration()
    report = {"when": datetime.now(timezone.utc).isoformat(), "provider": settings.llm_provider, "model": settings.judge_model_id,
              "dry_run": args.dry_run, "judge_weight": {"text_llm": cal["weights"]["text_llm"], "image_vlm": cal["weights"]["image_vlm"]},
              "caveat": "Tiny sample: indicative only. Not enough for calibration or confidence intervals.", "results": {}}
    for modality, n in (("text", args.n_text), ("image", args.n_image)):
        res = evaluate(modality, n, args.dry_run, args.seed, cal)
        res["summary"] = summarise(res)
        report["results"][modality] = res
        print(f"{modality}: scored {res['scored']}/{n}", f"(stopped: {res['stopped_early']})" if res["stopped_early"] else "")
        print("  ", json.dumps(res["summary"]))
        if res["stopped_early"]:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("dry_run.json" if args.dry_run else f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
