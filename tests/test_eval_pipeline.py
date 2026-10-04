"""End-to-end test of the offline evaluation chain on synthetic caches:
cache -> calibrate -> benchmark (ablations, thresholds tuned on calibration data) -> tables.
"""
import json
import sys

import numpy as np
import pytest

from agents.calibration import load_calibration, recalibrate, sigmoid
from agents.schemas import ToolVerdict
from eval import make_tables, run_benchmark
from eval.cache import append_line, load_cache, make_record
from training import calibrate


def write_cache(directory, n=400, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    for k in range(n):
        label = int(k % 2)
        sample = {"id": f"s{seed}_{k}", "label": label, "meta": {"generator": "gen_a" if k % 4 < 2 else "gen_b"}}
        sign = 2 * label - 1
        dl_logit = sign * (2.0 + shift) + rng.normal(0, 2.5)                      # overconfident -> temperature > 1
        p = sigmoid(dl_logit)
        dl = ToolVerdict(tool_name="text_dl", modality="text", score=p, confidence=abs(p - 0.5) * 2,
                         explanation="dl", raw_features={"raw_logit": float(dl_logit), "conf_factor": 1.0})
        bino = 0.95 - 0.12 * sign + rng.normal(0, 0.08)
        slm = ToolVerdict(tool_name="text_slm", modality="text", score=0.5, confidence=0.2,
                          explanation="slm", raw_features={"binoculars_score": float(bino), "conf_factor": 1.0})
        llm_score = float(np.clip(0.5 + sign * 0.25 + rng.normal(0, 0.2), 0.01, 0.99))
        llm = ToolVerdict(tool_name="text_llm", modality="text", score=llm_score, confidence=0.7, explanation="llm")
        for v in (dl, slm, llm):
            append_line(directory / f"{v.tool_name}.jsonl", make_record(sample, v, n_words=120))


@pytest.fixture
def caches(tmp_path):
    calib, test = tmp_path / "calib", tmp_path / "test"
    write_cache(calib, seed=1)
    write_cache(test, seed=2)
    return calib, test


def run_cli(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *map(str, argv)])
    module.main()


def test_calibration_fits_and_improves_reliability(caches, tmp_path, monkeypatch):
    calib, _ = caches
    out = tmp_path / "cal.json"
    run_cli(calibrate, ["--modality", "text", "--cache-dir", calib, "--output", out,
                        "--report-dir", tmp_path / "rep", "--plots-dir", tmp_path / "plots"], monkeypatch)
    cal = json.loads(out.read_text())
    assert cal["text_dl"]["temperature"] > 1.2                          # data was overconfident on purpose
    assert cal["text_slm"]["a"] < 0 and cal["text_slm"]["calibrated"] is True
    assert set(cal["weights"]) >= {"text_dl", "text_slm", "text_llm"} and all(w >= 0.05 for w in cal["weights"].values())
    th = cal["thresholds"]["text"]
    assert th["t_lo"] <= th["t_star"] <= th["t_hi"]
    report = json.loads((tmp_path / "rep" / "text_calibration_report.json").read_text())
    assert report["tools"]["text_dl"]["ece_after"] <= report["tools"]["text_dl"]["ece_before"]
    assert report["fused"]["auroc"] > 0.8


def test_recalibrate_follows_config_without_rerunning_models(caches, tmp_path, monkeypatch):
    calib, _ = caches
    cache = load_cache(calib, apply_calibration=False)
    v = next(iter(cache.samples.values()))["verdicts"]["text_dl"]
    cal = load_calibration()
    hot = json.loads(json.dumps(cal))
    hot["text_dl"]["temperature"] = 4.0
    assert abs(recalibrate(v, hot).score - 0.5) < abs(recalibrate(v, cal).score - 0.5)
    judge = next(iter(cache.samples.values()))["verdicts"]["text_llm"]
    assert recalibrate(judge, hot).score == judge.score                  # judge tools untouched


def test_benchmark_produces_all_ablation_rows_and_tables(caches, tmp_path, monkeypatch):
    calib, test = caches
    cal_path = tmp_path / "cal.json"
    run_cli(calibrate, ["--modality", "text", "--cache-dir", calib, "--output", cal_path,
                        "--report-dir", tmp_path / "rep", "--plots-dir", tmp_path / "plots"], monkeypatch)
    from config import settings
    monkeypatch.setattr(settings, "calibration_path", str(cal_path))

    out = tmp_path / "res"
    run_cli(run_benchmark, ["--modality", "text", "--cache-dir", test, "--tune-cache-dir", calib, "--name", "synth",
                            "--output-dir", out, "--bootstrap", 50], monkeypatch)
    res = json.loads((out / "results.json").read_text())
    assert res["n"] == 400
    assert set(res["configs"]) >= {"text_dl", "text_slm", "text_llm", "text_dl+text_slm", "text_dl+text_llm",
                                   "text_llm+text_slm", "full"}
    full = res["configs"]["full"]
    assert full["metrics"]["roc_auc"] > 0.9 and 0.0 <= full["threshold"] <= 1.0
    assert full["metrics"]["roc_auc"] >= max(res["configs"][t]["metrics"]["roc_auc"] for t in res["tools"]) - 0.02
    assert "selective" in full and 0 <= full["selective"]["coverage"] <= 1
    assert len(full["ci95"]["roc_auc"]) == 2 and "generator" in res["by_group"]
    assert len(res["per_sample"]) == 400 and "full" in res["per_sample"][0]["scores"]
    for f in ("metrics_table.csv", "summary.md", "plots/roc_all_configs.png", "plots/pr_all_configs.png",
              "plots/metrics_all_configs.png", "plots/full/confusion_matrix.png", "plots/full/calibration.png"):
        assert (out / f).exists(), f

    tables = tmp_path / "tables"
    run_cli(make_tables, ["--modality", "text", "--results", out / "results.json", "--out-dir", tables,
                          "--metrics", "roc_auc", "accuracy"], monkeypatch)
    assert "full" in (tables / "text_roc_auc.md").read_text() and (tables / "text_accuracy.tex").exists()


def test_reflexion_row_uses_rechecked_judge_verdicts(caches, tmp_path, monkeypatch):
    _, test = caches
    cache = load_cache(test)
    for i in list(cache.samples)[:40]:                                   # pretend the judge was re-run on 40 samples
        s = cache.samples[i]
        good = ToolVerdict(tool_name="text_llm", modality="text", score=0.95 if s["label"] else 0.05,
                           confidence=0.9, explanation="reflex", raw_features={"reflexion": True})
        append_line(test / "text_llm_reflex.jsonl", make_record({"id": i, "label": s["label"]}, good, 120))
    out = tmp_path / "res"
    run_cli(run_benchmark, ["--modality", "text", "--cache-dir", test, "--name", "synth", "--output-dir", out,
                            "--bootstrap", 0], monkeypatch)
    res = json.loads((out / "results.json").read_text())
    assert "full+reflexion" in res["configs"]
    assert res["configs"]["full+reflexion"]["metrics"]["roc_auc"] >= res["configs"]["full"]["metrics"]["roc_auc"] - 0.005
