"""Frozen CLIP ViT-L/14 embeddings + logistic-regression probe for deepfake detection.

Zero-shot generalisation test: the probe is fitted ONLY on the training mix (Deepfakes, FaceSwap, NeuralTextures, Celeb-DF,
WildDeepfake); Face2Face / FaceShifter are never seen.

    python eval/clip_probe.py extract            # embeddings -> eval/cache/clip/*.npz
    python eval/clip_probe.py fit                # fit + evaluate + save models/clip_probe.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

IMG = Path("data/processed/image")
CACHE = Path("eval/cache/clip")
MODEL = "openai/clip-vit-large-patch14"
EXT = {".jpg", ".jpeg", ".png"}
SETS = {   # name -> (folder, max images, role)
    "train": (IMG / "mixed/train", 24000, "fit"),
    "valid": (IMG / "mixed/valid", 4000, "fit"),
    "wd_test": (IMG / "wilddeepfake/test", 3000, "test"),
    "ffpp_seen": (IMG / "tests/ffpp_seen", 2000, "test"),
    "ffpp_unseen": (IMG / "tests/ffpp_unseen", 2000, "test"),
    "celebdf": (IMG / "tests/celebdf", 3000, "test"),
}


def list_files(folder: Path, limit: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for cls, y in (("real", 0), ("fake", 1)):
        items = [p for p in sorted((folder / cls).rglob("*")) if p.suffix.lower() in EXT]
        if len(items) > limit // 2:
            items = [items[i] for i in sorted(rng.choice(len(items), limit // 2, replace=False))]
        out += [(str(p), y) for p in items]
    return out


class Imgs(Dataset):
    def __init__(self, files, proc):
        self.files, self.proc = files, proc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p, y = self.files[i]
        with Image.open(p) as im:
            return self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0], y


def extract() -> None:
    from transformers import CLIPImageProcessor, CLIPModel
    dev = torch.device("cuda")
    proc = CLIPImageProcessor.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, (folder, limit, _) in SETS.items():
        out = CACHE / f"{name}.npz"
        if out.exists():
            continue
        files = list_files(folder, limit)
        feats, ys = [], []
        with torch.no_grad():
            for x, y in tqdm(DataLoader(Imgs(files, proc), batch_size=128, num_workers=6), desc=name):
                f = model.get_image_features(pixel_values=x.to(dev, torch.float16))
                f = f.pooler_output if hasattr(f, "pooler_output") else f
                feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
                ys += y.tolist()
        np.savez(out, x=np.concatenate(feats), y=np.array(ys), paths=np.array([p for p, _ in files]))
        print(name, len(ys))


def fit() -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from eval.metrics import best_threshold, binary_metrics
    L = {n: np.load(CACHE / f"{n}.npz") for n in SETS}
    best = None
    for C in (0.5, 2, 8, 32):                     # choose C on the valid split only (never on tests)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(L["train"]["x"], L["train"]["y"])
        auc = roc_auc_score(L["valid"]["y"], clf.predict_proba(L["valid"]["x"])[:, 1])
        print(f"C={C}: valid AUC {auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C)
    clf = LogisticRegression(C=best[1], max_iter=2000, class_weight="balanced").fit(L["train"]["x"], L["train"]["y"])
    thr = best_threshold(L["valid"]["y"], clf.predict_proba(L["valid"]["x"])[:, 1], "f1")
    rows = ["| set | n | AUC | acc@val-thr | F1@val-thr | TPR@5%FPR |", "|---|---|---|---|---|---|"]
    for n in ("wd_test", "ffpp_seen", "ffpp_unseen", "celebdf"):
        p = clf.predict_proba(L[n]["x"])[:, 1]
        m = binary_metrics(L[n]["y"], p, thr)
        rows.append(f"| {n} | {len(p)} | {m['roc_auc']:.3f} | {m['accuracy']:.3f} | {m['f1']:.3f} | {m['tpr_at_5pct_fpr']:.3f} |")
    print("\n".join(rows))
    Path("eval/results/image_clip_probe.md").write_text("\n".join(rows) + f"\nC={best[1]} threshold={thr:.3f}\n", encoding="utf-8")
    joblib.dump({"clf": clf, "threshold": thr, "model": MODEL}, "models/clip_probe.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit"])
    {"extract": extract, "fit": fit}[ap.parse_args().cmd]()
