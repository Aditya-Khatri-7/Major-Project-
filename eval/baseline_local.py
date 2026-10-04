"""Baseline: score the trained LOCAL models (no API) on every in-domain and cross-domain test set.

    python eval/baseline_local.py [tag] [text]  (model path via env TEXT_DL_MODEL_PATH)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from eval.metrics import binary_metrics  # noqa: E402

OUT = Path("eval/results")
TEXT_SETS = ["hc3_test", "mage_test", "raid_test_all", "raid_unseen_gen", "raid_unseen_attack"]
IMAGE_SETS = {"wilddeepfake_test": "data/processed/image/wilddeepfake/test", "ffpp_test": "data/processed/image/ffpp_test",
              "celebdf": "data/processed/image/celebdf"}
KEEP = ("roc_auc", "accuracy", "f1", "tpr_at_5pct_fpr", "ece")


def text_scores(rows):
    from agents.text_agent.tools.dl_classifier import window_logits
    out = []
    for r in tqdm(rows, leave=False):
        d = window_logits(r["text"])
        out.append(float(np.mean(d)) if d else 0.0)
    return 1 / (1 + np.exp(-np.array(out)))


def image_scores(paths):
    from agents.image_agent.tools.dl_classifier import _load
    ctx = _load()
    model, dev, idx, tf = ctx["model"], ctx["device"], ctx["index"], ctx["transform"]
    out = []
    for i in tqdm(range(0, len(paths), 64), leave=False):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(batch).float()
        out += torch.sigmoid(lg[:, idx["fake"]] - lg[:, idx["real"]]).cpu().tolist()
    return np.array(out)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline_local"
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    only_text = mode == "text"
    res = {}
    for name in ([] if mode == "image" else TEXT_SETS):
        df = pd.read_csv(f"data/processed/text/{name}.csv")
        p = text_scores(df.to_dict("records"))
        res[f"text/{name}"] = {k: v for k, v in binary_metrics(df["label"].values, p).items() if k in KEEP} | {"n": len(df)}
        print(name, res[f"text/{name}"], flush=True)
    for name, folder in ({} if only_text else IMAGE_SETS).items():
        paths, y = [], []
        for cls, lab in (("real", 0), ("fake", 1)):
            fs = sorted(q for q in Path(folder, cls).rglob("*") if q.suffix.lower() in {".png", ".jpg", ".jpeg"})
            paths += [str(q) for q in fs]
            y += [lab] * len(fs)
        p = image_scores(paths)
        res[f"image/{name}"] = {k: v for k, v in binary_metrics(np.array(y), p).items() if k in KEEP} | {"n": len(y)}
        print(name, res[f"image/{name}"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2))
    lines = ["| set | n | AUC | acc | F1 | TPR@5%FPR |", "|---|---|---|---|---|---|"]
    for k, m in res.items():
        lines.append(f"| {k} | {m['n']} | {m['roc_auc']:.3f} | {m['accuracy']:.3f} | {m['f1']:.3f} | {m['tpr_at_5pct_fpr']:.3f} |")
    (OUT / f"{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
