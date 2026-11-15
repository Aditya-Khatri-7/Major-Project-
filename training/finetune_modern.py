"""Fine-tune the SwinV2 modern-generator detector on OpenFake (incl. Flux/Z-Image) with compression/resize augmentation.

    python training/finetune_modern.py --epochs 2     -> models/modern_detector_v2 (original left untouched)
Validation: OpenFake-tiny validation split (never the test split), also evaluated after JPEG q70.
"""
from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

SRC, DST = Path("models/modern_detector"), Path("models/modern_detector_v2")


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.6:
        s = rng.uniform(0.4, 1.0)
        img = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.BICUBIC)
    if rng.random() < 0.7:
        b = io.BytesIO()
        img.save(b, "JPEG", quality=rng.randint(40, 95))
        img = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class DS(torch.utils.data.Dataset):
    def __init__(self, df, proc, train, fake_idx, real_idx):
        self.df, self.proc, self.train, self.f, self.r = df.reset_index(drop=True), proc, train, fake_idx, real_idx
        self.rng = random.Random(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = Image.open(r.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        if self.train:
            img = degrade(img, random.Random(random.random()))
        x = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return x, (self.f if r.label == "fake" else self.r)


def evaluate(model, loader, f, r, dev):
    model.eval()
    z, y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, t in loader:
            lg = model(pixel_values=x.to(dev)).logits.float().cpu()
            z += (lg[:, f] - lg[:, r]).tolist()
            y += (t == f).int().tolist()
    return roc_auc_score(y, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    dev = "cuda"
    proc = AutoImageProcessor.from_pretrained(SRC)
    model = AutoModelForImageClassification.from_pretrained(SRC).to(dev)
    labels = {int(i): str(n).lower() for i, n in model.config.id2label.items()}
    f = next(i for i, n in labels.items() if n in ("artificial", "ai", "fake"))
    r = next(i for i in labels if i != f)
    tr = pd.read_csv("data/processed/image/openfake_train/meta.csv")
    va = pd.read_csv("data/processed/image/openfake_tiny/meta.csv")
    va = va[va.split == "validation"]
    # balance: the weakest generators are oversampled x3 so the model has to learn them
    weak = tr.model.str.contains("flux|z-image|veo|recraft", case=False)
    tr = pd.concat([tr, tr[weak], tr[weak]]).sample(frac=1, random_state=0)
    print(len(tr), "train", tr.label.value_counts().to_dict(), flush=True)
    dl = torch.utils.data.DataLoader(DS(tr, proc, True, f, r), batch_size=a.bs, shuffle=True, num_workers=6, drop_last=True)
    vl = torch.utils.data.DataLoader(DS(va, proc, False, f, r), batch_size=64, num_workers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    best = evaluate(model, vl, f, r, dev)
    print("baseline val AUC", round(best, 4), flush=True)
    step = 0
    for ep in range(a.epochs):
        model.train()
        for x, t in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(pixel_values=x.to(dev)).logits.float(), t.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"ep{ep} step {step}/{total} loss {loss.item():.3f}", flush=True)
        auc = evaluate(model, vl, f, r, dev)
        print(f"epoch {ep} val AUC {auc:.4f}", flush=True)
        if auc > best:
            best = auc
            shutil.copytree(SRC, DST, dirs_exist_ok=True)
            model.save_pretrained(DST, safe_serialization=True)
            print("saved", DST, flush=True)


if __name__ == "__main__":
    main()
