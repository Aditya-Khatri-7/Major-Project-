"""Run each tool once over a dataset split and cache the verdicts (resumable, API-cost aware).

Text:   python eval/score_dataset.py --modality text --data data/processed/text/hc3_calib.csv \
            --tools dl,slm,llm --out-dir eval/cache/text_hc3_calib --limit 2000
Image:  python eval/score_dataset.py --modality image --data D:/archive/valid --subset calib \
            --tools dl,vlm --out-dir eval/cache/image_wd_calib --limit 2000

Re-running the same command skips samples already scored, so an interrupted run resumes and you can
raise --limit later. LLM / VLM tools run in parallel threads (--workers); local models run sequentially.
"""
from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from agents.common import IMAGE_TOOL_NAMES, TEXT_TOOL_NAMES  # noqa: E402
from agents.schemas import ToolVerdict  # noqa: E402
from eval.cache import append_line, done_ids, make_record  # noqa: E402
from eval.datasets import load_image_records, load_text_rows  # noqa: E402

ABORT_AFTER = 8          # abort if the first N verdicts of a tool are all errors (missing model / API key)
PARALLEL_TOOLS = {"llm", "vlm"}


def get_scorer(modality: str, key: str):
    if modality == "text":
        from agents.text_agent.tools import dl_classifier, llm_judge, slm_binoculars
        return {"dl": dl_classifier.score_text, "slm": slm_binoculars.score_text, "llm": llm_judge.score_text}[key]
    from agents.image_agent.tools import clip_probe, dl_classifier, general_probe, vlm_judge
    return {"dl": lambda p: dl_classifier.score_image(p, with_gradcam=False), "clip": clip_probe.score_image, "general": general_probe.score_image,
            "vlm": vlm_judge.score_image}[key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--data", type=Path, required=True, help="text: processed CSV | image: folder (real/, fake/) or CSV")
    ap.add_argument("--subset", choices=["all", "es", "calib"], default="all", help="image validation split to use")
    ap.add_argument("--tools", default=None, help="comma list; default dl,slm,llm (text) or dl,vlm (image)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="stratified sample size (controls API cost)")
    ap.add_argument("--workers", type=int, default=4, help="threads for LLM/VLM tools")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = TEXT_TOOL_NAMES if args.modality == "text" else IMAGE_TOOL_NAMES
    keys = [k.strip() for k in (args.tools or ",".join(names)).split(",") if k.strip()]
    unknown = [k for k in keys if k not in names]
    if unknown:
        ap.error(f"unknown tool(s) {unknown}; choose from {list(names)}")

    if args.modality == "text":
        samples = load_text_rows(args.data, args.limit, args.seed)
    else:
        samples = load_image_records(args.data, args.subset, args.limit, args.seed)
    print(f"{len(samples)} samples ({sum(s['label'] for s in samples)} positive) from {args.data}")

    for key in keys:
        tool_name = names[key]
        out_path = args.out_dir / f"{tool_name}.jsonl"
        finished = done_ids(out_path)
        todo = [s for s in samples if s["id"] not in finished]
        print(f"[{tool_name}] {len(finished)} cached, {len(todo)} to score")
        if not todo:
            continue
        scorer = get_scorer(args.modality, key)
        lock = threading.Lock()
        stats = {"n": 0, "errors": 0, "last_error": ""}

        def work(sample: dict) -> tuple[dict, ToolVerdict]:
            arg = sample["text"] if args.modality == "text" else sample["path"]
            try:
                verdict = scorer(arg)
            except Exception as exc:                       # a crashing tool becomes an error verdict
                verdict = ToolVerdict.failure(tool_name, args.modality, f"{type(exc).__name__}: {exc}")
            return sample, verdict

        def record(sample: dict, verdict: ToolVerdict) -> None:
            n_words = len(sample["text"].split()) if args.modality == "text" else 0
            with lock:
                append_line(out_path, make_record(sample, verdict, n_words))
                stats["n"] += 1
                if verdict.error:
                    stats["errors"] += 1
                    stats["last_error"] = verdict.explanation
                if stats["n"] == ABORT_AFTER and stats["errors"] == ABORT_AFTER:
                    raise SystemExit(f"[{tool_name}] first {ABORT_AFTER} verdicts all failed: {stats['last_error']}")

        if key in PARALLEL_TOOLS and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(work, s) for s in todo]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=tool_name):
                    record(*fut.result())
        else:
            for s in tqdm(todo, desc=tool_name):
                record(*work(s))
        print(f"[{tool_name}] done: {stats['n']} scored, {stats['errors']} errors")
    print(f"Cache: {args.out_dir}")


if __name__ == "__main__":
    main()
