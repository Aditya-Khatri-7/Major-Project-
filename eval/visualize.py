"""Regenerate the report plots from a benchmark results file (or training history) without re-running anything.

    python eval/visualize.py --results eval/results/text/hc3_test/results.json --config full --out-dir eval/plots/text/hc3_test
    python eval/visualize.py --history eval/results/image_dl/history.json --out-dir eval/plots/image_dl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.plots import generate_report_plots, plot_training_curves  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="results.json written by eval/run_benchmark.py")
    ap.add_argument("--config", default="full", help="configuration to plot from --results")
    ap.add_argument("--history", type=Path, help="history.json written by a training script")
    ap.add_argument("--name", default="model")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if not args.results and not args.history:
        ap.error("give --results and/or --history")

    if args.history:
        history = json.loads(args.history.read_text(encoding="utf-8"))
        best = None
        if "is_best" in history:
            flags = [e for e, b in zip(history["epoch"], history["is_best"]) if b]
            best = flags[-1] if flags else None
        plot_training_curves(history, args.out_dir, best, args.name)
        print(f"Training curves saved to {args.out_dir}")
    if args.results:
        data = json.loads(args.results.read_text(encoding="utf-8"))
        if args.config not in data["configs"]:
            raise SystemExit(f"Config '{args.config}' not in results; available: {list(data['configs'])}")
        y = [s["label"] for s in data["per_sample"]]
        p = [s["scores"][args.config] for s in data["per_sample"]]
        thr = data["configs"][args.config]["threshold"]
        generate_report_plots(y, p, args.out_dir, f"{data['name']}: {args.config}", thr)
        print(f"Report plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
