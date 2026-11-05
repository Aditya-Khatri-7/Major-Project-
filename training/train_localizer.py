"""Train a pixel-level edit localiser on real Stable Diffusion inpainting edits (eval/make_local_edits.py), with exact masks.

    python training/train_localizer.py --epochs 8
Model: DeepLabV3-ResNet50 (ImageNet backbone), one-channel output = P(pixel was AI-edited). Unedited originals are negatives (empty mask).
Train: data/processed/image/local_edits/train (90/10 train/val by id). Held-out test: .../test (different base photos).
Augmentation: flips, JPEG recompression, resize jitter, noise, so it does not rely on pristine pixels.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.models.segmentation import deeplabv3_resnet50  # noqa: E402

ROOT = Path("data/processed/image/local_edits")
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def build() -> nn.Module:
    m = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1, aux_loss=False)
    return m


class Edits(Dataset):
    def __init__(self, split: str, ids: list[str], train: bool):
        self.root, self.train = ROOT / split, train
        self.items = [(i, k) for i in ids for k in ("real", "edited")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        i, kind = self.items[n]
        img = Image.open(self.root / kind / f"{i}.png").convert("RGB")
        mask = Image.open(self.root / "mask" / f"{i}.png").convert("L") if kind == "edited" else Image.new("L", img.size, 0)
        if self.train:
            if random.random() < 0.5:
                img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:                                    # social-media style recompression
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=random.randint(60, 95))
                img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            if random.random() < 0.3:
                s = random.randint(384, 640)
                img = img.resize((s, s), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
        a = np.asarray(img, np.float32) / 255.0
        if self.train and random.random() < 0.25:
            a = np.clip(a + np.random.normal(0, random.uniform(0.002, 0.02), a.shape).astype(np.float32), 0, 1)
        x = torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1))
        y = torch.from_numpy((np.asarray(mask, np.float32) > 127).astype(np.float32))[None]
        return x, y


def dice_bce(logits, y):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)["out"]
        tot += float(dice_bce(out.float(), y)) * len(x)
        n += len(x)
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output", type=Path, default=Path("models/localizer_v1.pt"))
    a = ap.parse_args()
    ids = sorted(p.stem for p in (ROOT / "train" / "edited").glob("*.png"))
    random.Random(0).shuffle(ids)
    cut = int(len(ids) * 0.9)
    tr = DataLoader(Edits("train", ids[:cut], True), batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(Edits("train", ids[cut:], False), batch_size=a.batch_size, num_workers=2)
    device = torch.device("cuda")
    model = build().to(device)
    from torchvision.models import ResNet50_Weights, resnet50
    sd = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    print("backbone init:", model.backbone.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr))
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for k, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)["out"]
            loss = dice_bce(out.float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss)
            if k % 50 == 0:
                print(f"ep {ep + 1} step {k}/{len(tr)} loss {run / (k + 1):.4f}", flush=True)
        v = evaluate(model, va, device)
        print(f"== epoch {ep + 1}: train {run / len(tr):.4f} val {v:.4f}", flush=True)
        if v < best:
            best = v
            a.output.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), a.output)
            print("   saved best", flush=True)


if __name__ == "__main__":
    main()
