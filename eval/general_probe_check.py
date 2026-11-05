"""Does the general AI-image probe (whole image, CLIP + GRAVEX) catch face fakes the face tools miss?

    python eval/general_probe_check.py --n 150
Scores every held-out face set with the general probe, the CNN and the face CLIP probe, saves scores for fusion experiments
(eval/cache/general_check.json) and prints per-tool AUROC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.image_agent.face_crop import crop_face_if_needed  # noqa: E402
from agents.image_agent.tools import clip_probe, dl_classifier, general_probe  # noqa: E402
from eval.clip_face_v3 import TYPES, collect  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
args = ap.parse_args()
rng = np.random.default_rng(11)
wiki = collect("wiki")["test"]
real_dff = [str(p) for p in rng.choice(sorted(wiki.values()), args.n, replace=False)]
sets = {}
for t in TYPES:
    fk = collect(t)["test"]
    sets["dff_" + t] = (real_dff, [str(p) for p in rng.choice(sorted(fk.values()), args.n, replace=False)])
for name in ("celebdf", "ffpp_seen", "ffpp_unseen"):
    r = load_image_records(f"data/processed/image/tests/{name}", "all", 2 * args.n, 42)
    sets[name] = ([s["path"] for s in r if s["label"] == 0], [s["path"] for s in r if s["label"] == 1])
out, cache = {}, {}
for name, (re, fa) in sets.items():
    y = [0] * len(re) + [1] * len(fa)
    rows = []
    for p in re + fa:
        c = crop_face_if_needed(p) or p
        rows.append({"gen": general_probe.score_image(p).score, "dl": dl_classifier.score_image(c, with_gradcam=False).score,
                     "clip": clip_probe.score_image(c).score})
    cache[name] = {"y": y, "rows": rows}
    out[name] = {k: round(float(roc_auc_score(y, [r[k] for r in rows])), 3) for k in ("gen", "dl", "clip")}
    print(name, out[name], flush=True)
Path("eval/cache/general_check.json").write_text(json.dumps(cache))
