"""Unified CLIP face probe: face-swap deepfakes (cached embeddings) + AI-synthesised faces (GRAVEX, DeepDetect).

Both kinds are 'fake faces'; the existing probe only knew face swaps. Held-out tests per source; compares with the
current probe (models/clip_probe.joblib) on the face-swap sets.

    python eval/clip_face_unified.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from transformers import CLIPImageProcessor, CLIPModel  # noqa: E402

from eval.clip_general import EXT, MODEL, embed  # noqa: E402

CACHE = Path("eval/cache/clip")
GRAVEX = Path("data/raw/gravex/my_real_vs_ai_dataset/my_real_vs_ai_dataset")
DD = Path("data/raw/deepdetect/ddata")
# name -> (real dir, fake dir, n per class, split)
SYNTH = [
    ("gravex_train", GRAVEX / "real", GRAVEX / "ai_images", 6000, "train"),
    ("gravex_val", GRAVEX / "real", GRAVEX / "ai_images", 1000, "val"),
    ("gravex_test", GRAVEX / "real", GRAVEX / "ai_images", 1500, "test"),
    ("deepdetect_train", DD / "train/real", DD / "train/fake", 5000, "train"),
    ("deepdetect_val", DD / "train/real", DD / "train/fake", 1000, "val"),
    ("deepdetect_test", DD / "test/real", DD / "test/fake", 2000, "test"),
]


def synth_embeddings(model, proc, device):
    rng = np.random.default_rng(1)
    used, out = {}, {}
    for name, rd, fd, n, split in SYNTH:
        f = CACHE / f"{name}.npz"
        files, y = [], []
        for d, lab in ((rd, 0), (fd, 1)):
            items = sorted(q for q in d.iterdir() if q.suffix.lower() in EXT)
            taken = used.setdefault(str(d), set())
            pick = rng.choice([i for i in range(len(items)) if i not in taken], n, replace=False)
            taken.update(int(i) for i in pick)
            files += [str(items[i]) for i in pick]; y += [lab] * n
        if f.exists():
            d = np.load(f, allow_pickle=True); out[name] = (d["x"], d["y"], split); continue
        X, ok = embed(files, model, proc, device)
        X, y = X[ok], np.array(y)[ok]
        np.savez(f, x=X, y=y, paths=np.array(files)[ok])
        out[name] = (X, y, split)
    return out


def main():
    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(MODEL)
    synth = synth_embeddings(model, proc, device)
    swap = {k: np.load(CACHE / f"{k}.npz", allow_pickle=True) for k in ("train", "valid", "wd_test", "ffpp_seen", "ffpp_unseen", "celebdf")}
    Xtr = np.concatenate([swap["train"]["x"]] + [v[0] for v in synth.values() if v[2] == "train"])
    ytr = np.concatenate([swap["train"]["y"]] + [v[1] for v in synth.values() if v[2] == "train"])
    Xva = np.concatenate([swap["valid"]["x"]] + [v[0] for v in synth.values() if v[2] == "val"])
    yva = np.concatenate([swap["valid"]["y"]] + [v[1] for v in synth.values() if v[2] == "val"])
    best = None
    for C in (2, 8, 32):
        clf = LogisticRegression(C=C, max_iter=4000).fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.decision_function(Xva))
        print(f"C={C}: val AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    _, C, clf = best
    old = joblib.load("models/clip_probe.joblib")["clf"]
    print(f"{'test set':18} {'unified AUC':>11} {'unified acc':>11} | {'old probe AUC':>13}")
    tests = {k: (swap[k]["x"], swap[k]["y"]) for k in ("wd_test", "ffpp_seen", "ffpp_unseen", "celebdf")}
    tests.update({k: (v[0], v[1]) for k, v in synth.items() if v[2] == "test"})
    for k, (X, y) in tests.items():
        s = clf.decision_function(X)
        o = roc_auc_score(y, old.decision_function(X))
        print(f"{k:18} {roc_auc_score(y, s):11.3f} {accuracy_score(y, s > 0):11.3f} | {o:13.3f}", flush=True)
    joblib.dump({"clf": clf, "model": MODEL, "C": C}, "models/clip_face_unified.joblib")
    print("saved models/clip_face_unified.joblib")


if __name__ == "__main__":
    main()
