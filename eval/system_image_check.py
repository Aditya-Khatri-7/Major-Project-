"""Full image-system check (CNN + CLIP probe + frozen fusion/thresholds) on held-out face sets, including DeepFakeFace test faces.

    python eval/system_image_check.py --n 300
Reports fused AUROC, share of fakes / real faces called synthetic or authentic, escalation rate and accuracy on decided cases.
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
from agents.image_agent.tools import clip_probe, dl_classifier  # noqa: E402
from eval.clip_face_v3 import TYPES, collect  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("eval/results/system_image_check.json"))
    args = ap.parse_args()
    cal = load_calibration()
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
    out = {}
    for name, (re, fa) in sets.items():
        y = np.array([0] * len(re) + [1] * len(fa))
        rs = [fuse([dl_classifier.score_image(p, with_gradcam=False), clip_probe.score_image(p)], "image", 0, cal) for p in re + fa]
        f = np.array([0.5 if r.fused is None else r.fused for r in rs])
        ver = np.array([r.verdict for r in rs]); esc = np.array([r.escalate for r in rs])
        dec = ver != "uncertain"
        out[name] = {
            "auc": float(roc_auc_score(y, f)),
            "fakes_called_synthetic": float((ver[y == 1] == "synthetic").mean()),
            "fakes_called_authentic": float((ver[y == 1] == "authentic").mean()),
            "real_called_synthetic": float((ver[y == 0] == "synthetic").mean()),
            "real_called_authentic": float((ver[y == 0] == "authentic").mean()),
            "escalated": float(esc.mean()),
            "accuracy_when_decided": float(((ver[dec] == "synthetic") == (y[dec] == 1)).mean()) if dec.any() else None,
        }
        print(name, {k: round(v, 3) for k, v in out[name].items() if v is not None}, flush=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
