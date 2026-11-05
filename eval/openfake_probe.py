"""Modern-generator probe: CLIP ViT-L/14 embeddings of OpenFake images (Flux, Midjourney 6/7, GPT-image, SD3.5 ...) vs real photos.

    python eval/openfake_probe.py --embed          # embed data/processed/image/openfake_*/meta.csv (cached)
    python eval/openfake_probe.py                  # baseline of existing tools + fit new probe, report by split and generator
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

CACHE = Path("eval/cache/clip")
ap = argparse.ArgumentParser()
ap.add_argument("--embed", action="store_true")
ap.add_argument("--meta", default="data/processed/image/openfake_tiny/meta.csv")
ap.add_argument("--tag", default="openfake_tiny")
args = ap.parse_args()
meta = pd.read_csv(args.meta)
meta["y"] = (meta.label == "fake").astype(int)

if args.embed:
    from agents.image_agent.tools.clip_probe import _load
    ctx = _load()
    feats = []
    for i in range(0, len(meta), 64):
        xs = []
        for p in meta.path[i:i + 64]:
            try:
                with Image.open(p) as im:
                    xs.append(ctx["proc"](images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0])
            except Exception:
                xs.append(torch.zeros(3, 224, 224))
        with torch.no_grad():
            f = ctx["model"].get_image_features(pixel_values=torch.stack(xs).to(ctx["device"], ctx["dtype"]))
            f = f.pooler_output if hasattr(f, "pooler_output") else f
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        print(i, end=" ", flush=True)
    np.savez(CACHE / f"{args.tag}.npz", x=np.concatenate(feats), y=meta.y.values, split=meta.split.values, model=meta.model.values)
    sys.exit(0)

d = np.load(CACHE / f"{args.tag}.npz", allow_pickle=True)
X, y, sp, mo = d["x"], d["y"], d["split"], d["model"]
te = sp == "test"
old_gen = joblib.load("models/clip_general.joblib")["clf"]
old_face = joblib.load("models/clip_probe_v3.joblib")
old_face = old_face["clf"] if isinstance(old_face, dict) else old_face
print("test n", te.sum(), "fake share", y[te].mean().round(2))
print("existing general probe AUC", round(roc_auc_score(y[te], old_gen.decision_function(X[te])), 3))
print("existing face-CLIP v3  AUC", round(roc_auc_score(y[te], old_face.decision_function(X[te])), 3))
tr = sp != "test"
for C in (0.5, 2, 8, 32):
    clf = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
    print(f"new probe (OpenFake train+val only) C={C}: test AUC", round(roc_auc_score(y[te], clf.decision_function(X[te])), 3))
