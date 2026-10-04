"""Offline benchmark and ablation study from cached tool scores (no model or API calls).

    python eval/run_benchmark.py --modality text --cache-dir eval/cache/text_hc3_test \
        --tune-cache-dir eval/cache/text_hc3_calib --name hc3_test --output-dir eval/results/text/hc3_test

Configurations evaluated (all computed from the same cached verdicts):
    every single tool, every pair of tools, "full" (all tools fused), and "full+reflexion" when
    eval/run_reflexion.py has produced re-checked judge verdicts for the disagreement cases.

Decision thresholds are tuned per configuration on the tuning cache (calibration split) and then
frozen before touching the test cache. Without --tune-cache-dir a threshold of 0.5 is used and a
warning is printed. Outputs: results.json, metrics_table.csv, summary.md and plots.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.common import JUDGE_TOOLS  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from eval.cache import Cache, load_cache  # noqa: E402
from eval.metrics import (best_threshold, binary_metrics, bootstrap_ci, nan_to_none,  # noqa: E402
                          selective_metrics)
from eval.plots import (generate_report_plots, plot_metric_bars, plot_pr_multi, plot_roc_multi)  # noqa: E402

GROUP_KEYS = ("generator", "domain", "attack", "source", "dataset", "model")


def config_names(tools: list[str], has_reflexion: bool) -> list[str]:
    names = list(tools)
    if len(tools) > 2:
        names += ["+".join(pair) for pair in itertools.combinations(tools, 2)]
    if len(tools) > 1:
        names.append("full")
        if has_reflexion:
            names.append("full+reflexion")
    return names


def score_config(cache: Cache, ids: list[str], name: str, modality: str, cal: dict) -> tuple[np.ndarray, list[bool], list[str]]:
    """Return (P(synthetic) per sample, escalate flag per sample, verdict per sample) for one configuration."""
    reflexion = name == "full+reflexion"
    tools = cache.tools if name in ("full", "full+reflexion") else name.split("+")
    scores, escalate, verdicts = [], [], []
    for i in ids:
        s = cache.samples[i]
        chosen = [s["verdicts"][t] for t in tools]
        if reflexion:
            judge = JUDGE_TOOLS[modality]
            chosen = [cache.reflex.get(judge, {}).get(i, v) if v.tool_name == judge else v for v in chosen]
        if len(chosen) == 1:                                   # single tool: its own calibrated score
            v = chosen[0]
            scores.append(0.5 if v.error else v.score)
            escalate.append(v.error)
            verdicts.append("uncertain" if v.error else ("synthetic" if v.score >= 0.5 else "authentic"))
        else:
            r = fuse(chosen, modality, s["n_words"], cal, retries=1 if reflexion else 0)
            scores.append(0.5 if r.fused is None else r.fused)
            escalate.append(r.escalate)
            verdicts.append(r.verdict)
    return np.array(scores), escalate, verdicts


def _fast_metric(name: str, threshold: float):
    if name == "roc_auc":
        return lambda y, p: float(roc_auc_score(y, p))
    if name == "accuracy":
        return lambda y, p: float(((p >= threshold).astype(int) == y).mean())
    return lambda y, p: float(f1_score(y, (p >= threshold).astype(int), zero_division=0))


def group_breakdown(cache: Cache, ids: list[str], scores: np.ndarray, threshold: float) -> dict:
    out: dict = {}
    for key in GROUP_KEYS:
        groups: dict[str, list[int]] = {}
        for pos, i in enumerate(ids):
            value = cache.samples[i]["meta"].get(key)
            if value:
                groups.setdefault(str(value), []).append(pos)
        if len(groups) < 2:
            continue
        out[key] = {}
        for value, idx in sorted(groups.items()):
            y = np.array([cache.samples[ids[p]]["label"] for p in idx])
            p_ = scores[idx]
            entry = {"n": len(idx), "positive_rate": float(y.mean()),
                     "accuracy": float(((p_ >= threshold).astype(int) == y).mean())}
            if len(set(y.tolist())) == 2:
                entry["roc_auc"] = float(roc_auc_score(y, p_))
            out[key][value] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--cache-dir", type=Path, required=True, help="score cache of the TEST split")
    ap.add_argument("--tune-cache-dir", type=Path, default=None, help="score cache of the calibration split (thresholds)")
    ap.add_argument("--name", required=True, help="label for this test set, e.g. hc3_test")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 = off)")
    ap.add_argument("--rag", action="store_true", help="also measure retrieval coverage / citations for the full system")
    args = ap.parse_args()

    cal = load_calibration()
    test = load_cache(args.cache_dir, cal=cal)
    ids = test.ids()
    if not ids:
        raise SystemExit("No sample has a verdict from every tool; score all tools on the same samples.")
    y = np.array(test.labels(ids))
    print(f"Test set '{args.name}': {len(ids)} samples ({y.mean():.1%} positive); tools: {test.tools}")
    tune: Optional[Cache] = None
    if args.tune_cache_dir:
        tune = load_cache(args.tune_cache_dir, cal=cal)
    else:
        print("WARNING: no --tune-cache-dir; using threshold 0.5 for every configuration.")

    has_reflexion = bool(test.reflex.get(JUDGE_TOOLS[args.modality]))
    names = config_names(test.tools, has_reflexion)
    results: dict = {}
    per_config_scores: dict[str, np.ndarray] = {}
    per_config_esc: dict[str, list[bool]] = {}
    for name in names:
        scores, esc, verdicts = score_config(test, ids, name, args.modality, cal)
        threshold = 0.5
        if tune is not None:
            tune_ids = tune.ids()
            if name == "full+reflexion":
                base = "full"
            else:
                base = name
            t_scores, _, _ = score_config(tune, tune_ids, base, args.modality, cal)
            threshold = best_threshold(tune.labels(tune_ids), t_scores, "f1")
        m = binary_metrics(y, scores, threshold)
        entry: dict = {"metrics": m, "threshold": threshold}
        if args.bootstrap:
            entry["ci95"] = {k: bootstrap_ci(y, scores, _fast_metric(k, threshold), args.bootstrap)
                             for k in ("roc_auc", "accuracy", "f1")}
        if "+" in name or name.startswith("full"):
            pred = (scores >= threshold).astype(int)
            entry["selective"] = selective_metrics(y, pred, esc)
            entry["verdict_counts"] = {v: verdicts.count(v) for v in ("authentic", "synthetic", "uncertain")}
        results[name] = entry
        per_config_scores[name], per_config_esc[name] = scores, esc
        print(f"  {name:<32} AUC={m['roc_auc']:.4f} acc={m['accuracy']:.4f} F1={m['f1']:.4f} (thr {threshold:.3f})")

    for tool in test.tools:
        results.setdefault(tool, {})["errors"] = int(sum(test.samples[i]["verdicts"][tool].error for i in ids))

    headline = "full+reflexion" if has_reflexion else ("full" if "full" in results else test.tools[0])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    generate_report_plots(y, per_config_scores[headline], out / "plots" / headline.replace("+", "_"),
                          f"{args.name}: {headline}", results[headline]["threshold"])
    curves = {n: (y, per_config_scores[n]) for n in names}
    plot_roc_multi(curves, out / "plots" / "roc_all_configs.png", f"ROC - {args.name}")
    plot_pr_multi(curves, out / "plots" / "pr_all_configs.png", f"Precision-recall - {args.name}")
    plot_metric_bars({n: results[n]["metrics"] for n in names}, out / "plots" / "metrics_all_configs.png",
                     title=f"Ablation - {args.name}")

    payload = {
        "name": args.name, "modality": args.modality, "n": len(ids), "tools": test.tools,
        "calibration": {"weights": cal["weights"], "thresholds": cal["thresholds"][args.modality]},
        "configs": results,
        "by_group": group_breakdown(test, ids, per_config_scores[headline], results[headline]["threshold"]),
        "per_sample": [{"id": i, "label": int(y[k]), "scores": {n: float(per_config_scores[n][k]) for n in names}}
                       for k, i in enumerate(ids)],
    }

    if args.rag:
        from rag.retriever import build_query, retrieve_evidence
        covered, n_cites, sims = 0, 0, []
        for i in ids:
            s = test.samples[i]
            hits = retrieve_evidence(build_query(args.modality, list(s["verdicts"].values())), args.modality)
            covered += bool(hits)
            n_cites += len({h.source_id for h in hits})
            sims += [h.similarity_score for h in hits]
        payload["rag"] = {"citation_coverage": covered / len(ids), "mean_citations": n_cites / len(ids),
                          "mean_similarity": float(np.mean(sims)) if sims else None}
        print(f"  RAG: coverage={payload['rag']['citation_coverage']:.2%} mean citations={payload['rag']['mean_citations']:.2f}")

    (out / "results.json").write_text(json.dumps(nan_to_none(payload), indent=2), encoding="utf-8")
    with open(out / "metrics_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "tpr_at_1pct_fpr", "ece"]
        w.writerow(["config", "threshold"] + cols)
        for n in names:
            w.writerow([n, f"{results[n]['threshold']:.4f}"] + [f"{results[n]['metrics'][c]:.4f}" for c in cols])
    lines = [f"# {args.name} ({args.modality}, n={len(ids)})", "", "| config | acc | prec | recall | F1 | AUROC (95% CI) |",
             "|---|---|---|---|---|---|"]
    for n in names:
        m = results[n]["metrics"]
        ci = results[n].get("ci95", {}).get("roc_auc")
        ci_txt = f" ({ci[0]:.3f}-{ci[1]:.3f})" if ci and ci[0] == ci[0] else ""
        lines.append(f"| {n} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f}{ci_txt} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
