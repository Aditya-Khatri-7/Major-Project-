"""Offline Reflexion study: re-run the judge (LLM / VLM) only on the samples where the tools disagree.

    python eval/run_reflexion.py --modality text --data data/processed/text/hc3_test.csv \
        --cache-dir eval/cache/text_hc3_test [--no-rag] [--workers 4]

For every cached sample whose full-system spread exceeds the disagreement threshold, the judge is
called once more with the other tools' outputs (and retrieved notes, unless --no-rag). Results go to
<cache-dir>/<judge>_reflex.jsonl; eval/run_benchmark.py then reports an extra "full+reflexion" row.
Resumable: samples already re-checked are skipped.
"""
from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.common import JUDGE_TOOLS  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from agents.reflexion import evidence_summary, peer_summary  # noqa: E402
from eval.cache import REFLEX_SUFFIX, append_line, done_ids, load_cache, make_record  # noqa: E402
from eval.datasets import load_image_records, load_text_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--data", type=Path, required=True, help="the same data source used for score_dataset.py")
    ap.add_argument("--subset", choices=["all", "es", "calib"], default="all")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--no-rag", action="store_true", help="do not add retrieved notes to the judge prompt")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cal = load_calibration()
    judge = JUDGE_TOOLS[args.modality]
    cache = load_cache(args.cache_dir, cal=cal)
    if judge not in cache.tools:
        raise SystemExit(f"No cached '{judge}' scores in {args.cache_dir}; score the judge tool first.")

    data = (load_text_rows(args.data) if args.modality == "text"
            else load_image_records(args.data, args.subset))
    by_id = {r["id"]: r for r in data}

    out_path = args.cache_dir / f"{judge}{REFLEX_SUFFIX}.jsonl"
    finished = done_ids(out_path)
    todo = []
    for i in cache.ids():
        s = cache.samples[i]
        if i in finished or i not in by_id or s["verdicts"][judge].error:
            continue
        r = fuse(list(s["verdicts"].values()), args.modality, s["n_words"], cal)
        if r.needs_reflexion:
            todo.append(i)
    print(f"{len(todo)} disagreement samples to re-check ({len(finished)} already done).")
    if not todo:
        return

    if args.modality == "text":
        from agents.text_agent.tools import llm_judge as judge_mod
        rerun = judge_mod.score_text_with_context
    else:
        from agents.image_agent.tools import vlm_judge as judge_mod
        rerun = judge_mod.score_image_with_context

    def retrieve(i: str):
        if args.no_rag:
            return []
        from rag.retriever import build_query, retrieve_evidence
        return retrieve_evidence(build_query(args.modality, list(cache.samples[i]["verdicts"].values())), args.modality)

    lock = threading.Lock()

    def work(i: str):
        s, row = cache.samples[i], by_id[i]
        peers = peer_summary(list(s["verdicts"].values()), judge)
        notes = evidence_summary(retrieve(i))
        content = row["text"] if args.modality == "text" else row["path"]
        return i, rerun(content, peers, notes)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, i) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="reflexion"):
            i, verdict = fut.result()
            with lock:
                append_line(out_path, make_record({"id": i, "label": cache.samples[i]["label"],
                                                   "meta": cache.samples[i]["meta"]}, verdict,
                                                  cache.samples[i]["n_words"]))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
