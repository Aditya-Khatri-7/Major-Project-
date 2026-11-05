"""Fuse cached agent scores under the current calibration and report system-level metrics per held-out set.

    python eval/agent_image_eval.py [--tools image_dl,image_clip,image_modern]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from eval.cache import load_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tools", default="image_dl,image_clip,image_modern,image_edit")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--glob", default="agent_test_*")
a = ap.parse_args()
keep = set(a.tools.split(","))
cal = load_calibration()
res = {}
for d in sorted(Path("eval/cache").glob(a.glob)):
    c = load_cache(d, cal=cal)
    ids = list(c.samples)
    y = np.array([c.samples[i]["label"] for i in ids])
    rs = [fuse([v for t, v in c.samples[i]["verdicts"].items() if t in keep], "image", 0, cal) for i in ids]
    f = np.array([0.5 if r.fused is None else r.fused for r in rs])
    ver = np.array([r.verdict for r in rs])
    esc = np.array([r.escalate for r in rs])
    dec = ver != "uncertain"
    row = {"n": len(ids), "auc": float(roc_auc_score(y, f)),
           "fakes_called_authentic": float((ver[y == 1] == "authentic").mean()),
           "fakes_called_synthetic": float((ver[y == 1] == "synthetic").mean()),
           "real_called_synthetic": float((ver[y == 0] == "synthetic").mean()),
           "escalated": float(esc.mean()),
           "silent_wrong": float((((ver == "synthetic") & (y == 0)) | ((ver == "authentic") & (y == 1)))[~esc].mean() if (~esc).any() else 0),
           "accuracy_when_decided": float(((ver[dec] == "synthetic") == (y[dec] == 1)).mean()) if dec.any() else None}
    for t in sorted(keep):
        s = [c.samples[i]["verdicts"][t].score if t in c.samples[i]["verdicts"] and not c.samples[i]["verdicts"][t].error else 0.5 for i in ids]
        row["auc_" + t] = float(roc_auc_score(y, s))
    res[d.name] = row
    print(f"{d.name:28s}", {k: round(v, 3) for k, v in row.items() if v is not None}, flush=True)
if a.out:
    a.out.write_text(json.dumps(res, indent=2))
