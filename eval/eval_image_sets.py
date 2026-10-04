"""Evaluate the image classifier on folder-style test sets (real/ + fake/) and write a markdown/JSON table.

    python eval/eval_image_sets.py --weights models/efficientnet_b4.pt --name image_v3 \
        --set wd_test=data/processed/image/wilddeepfake/test \
        --set ffpp_seen=data/processed/image/tests/ffpp_seen ...

Threshold: the F1-optimal threshold from `--threshold-from` (a validation folder), never from the test set itself.
Reports AUC, accuracy, F1, TPR@5%FPR at 0.5 and at that validation threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from agents.image_agent.tools.dl_classifier import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402
from eval.datasets import IMAGE_EXTENSIONS, image_subset  # noqa: E402
from eval.metrics import best_threshold, binary_metrics  # noqa: E402
from torchvision import transforms  # noqa: E402

TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class Folder(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p, y = self.files[i]
        with Image.open(p) as im:
            return TF(im.convert("RGB")), y


def collect(folder: Path, limit: int | None, seed: int = 0, subset: str | None = None):
    """label 1 = fake (positive class)."""
    rng = np.random.default_rng(seed)
    out = {0: [], 1: []}
    for cls, y in (("real", 0), ("fake", 1)):
        for p in sorted((folder / cls).rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if subset and image_subset(p.relative_to(folder).as_posix()) != subset:
                continue
            out[y].append((str(p), y))
    files = []
    for y in (0, 1):
        items = out[y]
        if limit and len(items) > limit // 2:
            idx = rng.choice(len(items), limit // 2, replace=False)
            items = [items[i] for i in sorted(idx)]
        files += items
    return files


@torch.no_grad()
def score(model, files, device, batch=64, workers=4):
    loader = DataLoader(Folder(files), batch_size=batch, num_workers=workers)
    probs, ys = [], []
    for x, y in tqdm(loader, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        probs += torch.softmax(logits.float(), 1)[:, 0].cpu().tolist()      # class 0 = fake
        ys += y.tolist()
    return np.array(ys), np.array(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("models/efficientnet_b4.pt"))
    ap.add_argument("--name", default="image")
    ap.add_argument("--set", action="append", required=True, help="name=folder (with real/ and fake/)")
    ap.add_argument("--threshold-from", type=Path, default=None,
                    help="validation folder used ONLY to pick the F1-optimal threshold (its calib 40 percent)")
    ap.add_argument("--limit", type=int, default=4000, help="max images per set (balanced)")
    ap.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    thr = 0.5
    if args.threshold_from:
        y, p = score(model, collect(args.threshold_from, 4000, subset="calib"), device)
        thr = float(best_threshold(y, p, "f1"))
        print(f"Validation-calibration threshold: {thr:.3f}")

    results = {"threshold": thr, "weights": str(args.weights), "sets": {}}
    rows = ["| set | n | AUC | acc@0.5 | acc@val-thr | F1@val-thr | TPR@5%FPR |", "|---|---|---|---|---|---|---|"]
    for spec in args.set:
        name, folder = spec.split("=", 1)
        files = collect(Path(folder), args.limit)
        y, p = score(model, files, device)
        m05, mt = binary_metrics(y, p, 0.5), binary_metrics(y, p, thr)
        results["sets"][name] = {"n": len(y), "at_0.5": m05, "at_val_threshold": mt}
        rows.append(f"| {name} | {len(y)} | {m05['roc_auc']:.3f} | {m05['accuracy']:.3f} | {mt['accuracy']:.3f} | "
                    f"{mt['f1']:.3f} | {m05['tpr_at_5pct_fpr']:.3f} |")
        print(rows[-1])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (args.out_dir / f"{args.name}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
