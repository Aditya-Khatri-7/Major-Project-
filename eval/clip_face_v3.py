"""Face probe v3: v2 training data + DeepFakeFace (diffusion inpainting edits, face swaps, text-to-image faces).

    python eval/clip_face_v3.py --extract        # unzip data/raw/deepfakeface/*.zip once
    python eval/clip_face_v3.py                  # embed, fit, compare with the current probe, save models/clip_probe_v3.joblib

Splits are by file stem (the same stem is the same source face in wiki / inpainting / insight / text2img), so no face
appears in both train and test. Test sets of the earlier probe are reused unchanged for the regression check.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import CLIPImageProcessor, CLIPModel  # noqa: E402

from eval.clip_general import EXT, MODEL, embed  # noqa: E402

RAW = Path("data/raw/deepfakeface")
CACHE = Path("eval/cache/clip")
TYPES = ["inpainting", "insight", "text2img"]
OLD_TESTS = ("wd_test", "ffpp_seen", "ffpp_unseen", "celebdf", "gravex_test", "deepdetect_test")
N = {"train": 2000, "val": 350, "test": 500}          # per fake type; real = 3x this


def split_of(stem: str) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 80 else "test"


def extract() -> None:
    for z in sorted(RAW.glob("*.zip")):
        out = RAW / z.stem
        if out.exists():
            continue
        print("extracting", z.name, flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(RAW)


def collect(name: str) -> dict[str, dict[str, Path]]:
    by_split: dict[str, dict[str, Path]] = {"train": {}, "val": {}, "test": {}}
    for p in sorted((RAW / name).rglob("*")):
        if p.suffix.lower() in EXT:
            by_split[split_of(p.stem)][p.stem] = p
    return by_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    if args.extract:
        extract()
        return

    rng = np.random.default_rng(3)
    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(MODEL)

    wiki, fakes = collect("wiki"), {t: collect(t) for t in TYPES}
    new: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sp, n in N.items():
        files, y, kind = [], [], []
        real_pool = sorted(wiki[sp].values())
        for p in rng.choice(real_pool, min(len(real_pool), 3 * n), replace=False):
            files.append(str(p)); y.append(0); kind.append("real")
        for t in TYPES:
            pool = sorted(fakes[t][sp].values())
            for p in rng.choice(pool, min(len(pool), n), replace=False):
                files.append(str(p)); y.append(1); kind.append(t)
        cache = CACHE / f"dff_{sp}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            X, y, kind = d["x"], d["y"], d["kind"]
        else:
            X, ok = embed(files, model, proc, device)
            X, y, kind = X[ok], np.array(y)[ok], np.array(kind)[ok]
            np.savez(cache, x=X, y=y, kind=kind)
        new[sp] = (X, y, kind)
        print(sp, X.shape, flush=True)

    swap = {k: np.load(CACHE / f"{k}.npz", allow_pickle=True) for k in ("train", "valid", *OLD_TESTS)}
    old_tr = [swap["train"]] + [np.load(CACHE / f"{k}.npz", allow_pickle=True) for k in ("gravex_train", "deepdetect_train")]
    old_va = [swap["valid"]] + [np.load(CACHE / f"{k}.npz", allow_pickle=True) for k in ("gravex_val", "deepdetect_val")]
    Xtr = np.concatenate([d["x"] for d in old_tr] + [new["train"][0]])
    ytr = np.concatenate([d["y"] for d in old_tr] + [new["train"][1]])
    Xva = np.concatenate([d["x"] for d in old_va] + [new["val"][0]])
    yva = np.concatenate([d["y"] for d in old_va] + [new["val"][1]])

    best = None
    for C in (2, 8, 32):
        clf = LogisticRegression(C=C, max_iter=4000).fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.decision_function(Xva))
        print(f"C={C}: val AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    _, C, clf = best
    cur = joblib.load("models/clip_probe_v2.joblib")["clf"]

    print(f"\n{'test set':26} {'v3 AUC':>7} | {'v2 AUC':>7}")
    for k in OLD_TESTS:
        X, y = swap[k]["x"], swap[k]["y"]
        print(f"{k:26} {roc_auc_score(y, clf.decision_function(X)):7.3f} | {roc_auc_score(y, cur.decision_function(X)):7.3f}")
    X, y, kind = new["test"]
    real = kind == "real"
    for t in TYPES:
        m = real | (kind == t)
        print(f"{'dff_' + t:26} {roc_auc_score(y[m], clf.decision_function(X[m])):7.3f} | {roc_auc_score(y[m], cur.decision_function(X[m])):7.3f}")
    joblib.dump({"clf": clf, "model": MODEL, "C": C}, "models/clip_probe_v3.joblib")
    print("saved models/clip_probe_v3.joblib")


if __name__ == "__main__":
    main()
