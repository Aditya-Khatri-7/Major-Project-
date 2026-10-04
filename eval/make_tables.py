"""Combine several benchmark results into the paper's headline table (markdown, CSV and LaTeX).

    python eval/make_tables.py --modality text \
        --results eval/results/text/hc3_test/results.json eval/results/text/mage_test/results.json \
                  eval/results/text/raid_crossgen/results.json eval/results/text/raid_adv/results.json \
        --out-dir eval/results/tables --metrics roc_auc accuracy f1

Rows are configurations (each tool alone, pairs, full, full+reflexion); columns are test sets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.plots import plot_metric_bars  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("eval/results/tables"))
    ap.add_argument("--metrics", nargs="+", default=["roc_auc", "accuracy", "f1"])
    args = ap.parse_args()

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in args.results]
    names = [r["name"] for r in runs]
    configs: list[str] = []
    for r in runs:
        configs += [c for c in r["configs"] if "metrics" in r["configs"][c] and c not in configs]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        rows = []
        for c in configs:
            row = [c]
            for r in runs:
                m = r["configs"].get(c, {}).get("metrics", {}).get(metric)
                row.append("-" if m is None else f"{m:.3f}")
            rows.append(row)
        stem = f"{args.modality}_{metric}"
        with open(args.out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method"] + names)
            w.writerows(rows)
        md = ["| method | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        md += ["| " + " | ".join(r) + " |" for r in rows]
        (args.out_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        tex = ["\\begin{tabular}{l" + "c" * len(names) + "}", "\\hline",
               "Method & " + " & ".join(n.replace("_", "\\_") for n in names) + " \\\\ \\hline"]
        tex += [r[0].replace("_", "\\_") + " & " + " & ".join(r[1:]) + " \\\\" for r in rows]
        tex += ["\\hline", "\\end{tabular}"]
        (args.out_dir / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(f"\n{metric}:\n" + "\n".join(md))

    for r in runs:                                                   # per-test-set comparison bars
        plot_metric_bars({c: v["metrics"] for c, v in r["configs"].items() if "metrics" in v},
                         args.out_dir / f"{args.modality}_{r['name']}_bars.png", title=f"{r['name']}")
    print(f"\nTables written to {args.out_dir}")


if __name__ == "__main__":
    main()
