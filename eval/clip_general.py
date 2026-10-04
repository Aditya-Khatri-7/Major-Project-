"""General AI-generated-image detector: frozen CLIP ViT-L/14 + logistic regression, trained on DeepDetect-2025 + a real-vs-AI art set
(any content). Complements the face-deepfake tools, which cannot judge illustrations / AI art.

    python eval/clip_general.py            # sample, embed, fit, test, save models/clip_general.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import CLIPImageProcessor, CLIPModel  # noqa: E402

MODEL = "openai/clip-vit-large-patch14"
EXT = {".jpg", ".jpeg", ".png", ".webp"}


class Imgs(Dataset):
    def __init__(self, files, proc):
        self.files, self.proc = files, proc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            with Image.open(self.files[i]) as im:
                return self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0], 1
        except Exception:
            return torch.zeros(3, 224, 224), 0


def embed(files, model, proc, device):
    out, ok = [], []
    for x, good in tqdm(DataLoader(Imgs(files, proc), batch_size=64, num_workers=4), leave=False):
        with torch.no_grad():
            f = model.get_image_features(pixel_values=x.to(device, torch.float16))
            f = f.pooler_output if hasattr(f, "pooler_output") else f
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        ok += good.tolist()
    return np.concatenate(out), np.array(ok, bool)


DD = Path("data/raw/deepdetect/ddata")
ART = Path("data/raw/aiart/Data")
# (name, real dir, fake dir, n_per_class, split)
PLAN = [
    ("deepdetect", DD / "train/real", DD / "train/fake", 6000, "train"),
    ("deepdetect", DD / "train/real", DD / "train/fake", 1500, "val"),
    ("aiart", ART / "REAL", ART / "FAKE", 4000, "train"),
    ("aiart", ART / "REAL", ART / "FAKE", 1000, "val"),
    ("deepdetect_test", DD / "test/real", DD / "test/fake", 2000, "test"),
    ("aiart_test", ART / "REAL", ART / "FAKE", 1500, "test"),
]


def main():
    rng = np.random.default_rng(0)
    files, labels, split, src, used = [], [], [], [], {}
    for name, rd, fd, n, sp in PLAN:
        for d, y in ((rd, 0), (fd, 1)):
            items = sorted(q for q in d.iterdir() if q.suffix.lower() in EXT)
            taken = used.setdefault(str(d), set())
            avail = [i for i in range(len(items)) if i not in taken]
            pick = rng.choice(avail, n, replace=False)           # disjoint across train/val/test of the same folder
            taken.update(int(i) for i in pick)
            files += [str(items[i]) for i in pick]
            labels += [y] * n; split += [sp] * n; src += [name] * n
    labels, split, src = np.array(labels), np.array(split), np.array(src)
    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(MODEL)
    X, ok = embed(files, model, proc, device)
    print("unreadable images skipped:", int((~ok).sum()))
    tr, va = (split == "train") & ok, (split == "val") & ok
    best = None
    for C in (0.5, 2, 8, 32):
        clf = LogisticRegression(C=C, max_iter=3000).fit(X[tr], labels[tr])
        auc = roc_auc_score(labels[va], clf.decision_function(X[va]))
        print(f"C={C}: val AUC {auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    _, C, clf = best
    for name in ("deepdetect_test", "aiart_test"):
        te = (src == name) & (split == "test") & ok
        s = clf.decision_function(X[te])
        print(f"TEST {name} (n={int(te.sum())}): AUC {roc_auc_score(labels[te], s):.4f}, acc {accuracy_score(labels[te], s > 0):.4f}")
    joblib.dump({"clf": clf, "model": MODEL, "C": C}, "models/clip_general.joblib")
    print("saved models/clip_general.joblib")


if __name__ == "__main__":
    main()
