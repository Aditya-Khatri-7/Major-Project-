import csv
import json
import sys
from pathlib import Path

from PIL import Image

from data import prepare_text
from eval.datasets import CALIB_PERCENT, image_subset, load_image_records, load_text_rows


def make_hc3(path: Path, n_questions=200):
    with open(path, "w", encoding="utf-8") as f:
        for q in range(n_questions):
            f.write(json.dumps({
                "question": f"question number {q}?",
                "human_answers": [f"human answer {q} " + "word " * 30],
                "chatgpt_answers": [f"chatgpt answer {q} " + "token " * 30, "too short"],
                "source": "reddit_eli5" if q % 2 else "finance",
            }) + "\n")


def test_hc3_split_by_question_has_no_leakage(tmp_path, monkeypatch):
    src, out = tmp_path / "all.jsonl", tmp_path / "out"
    make_hc3(src)
    monkeypatch.setattr(sys, "argv", ["prog", "hc3", "--hc3-path", str(src), "--out-dir", str(out)])
    prepare_text.main()
    groups, total = {}, 0
    for split in ("train", "val", "calib", "test"):
        with open(out / f"hc3_{split}.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        groups[split] = {r["group"] for r in rows}
        total += len(rows)
        assert {int(r["label"]) for r in rows} <= {0, 1}
        assert all(len(r["text"].split()) >= 20 for r in rows)                  # "too short" answers dropped
    names = list(groups)
    assert all(not (groups[a] & groups[b]) for i, a in enumerate(names) for b in names[i + 1:])
    assert total == 400 and len(groups["train"]) > len(groups["test"]) > 0


def test_load_text_rows_stratified_limit(tmp_path):
    p = tmp_path / "x.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "generator"])
        w.writeheader()
        for i in range(100):
            w.writerow({"id": i, "text": f"t{i}", "label": int(i % 10 == 0), "generator": "g"})
    rows = load_text_rows(p, limit=20)
    assert len(rows) == 20 and sum(r["label"] for r in rows) == 10 and rows[0]["meta"] == {"generator": "g"}


def test_image_validation_split_is_deterministic_and_disjoint(tmp_path):
    for cls in ("real", "fake"):
        (tmp_path / cls).mkdir()
        for i in range(200):
            Image.new("RGB", (8, 8)).save(tmp_path / cls / f"{i}.png")
    es = {r["id"] for r in load_image_records(tmp_path, "es")}
    calib = {r["id"] for r in load_image_records(tmp_path, "calib")}
    everything = {r["id"] for r in load_image_records(tmp_path, "all")}
    assert es | calib == everything and not (es & calib)
    assert abs(len(calib) / len(everything) * 100 - CALIB_PERCENT) < 8
    assert image_subset("real/1.png") == image_subset("real\\1.png")
    labels = {r["id"]: r["label"] for r in load_image_records(tmp_path, "all")}
    assert labels["fake/3.png"] == 1 and labels["real/3.png"] == 0
    assert len(load_image_records(tmp_path, "all", limit=50)) == 50
