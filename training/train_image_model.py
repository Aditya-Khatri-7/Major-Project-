"""Fine-tune EfficientNet-B4 (ImageNet-pretrained) for real-vs-fake face image classification.

Dataset layout (WildDeepfake style):
    DATA_DIR/train/{fake,real}/   DATA_DIR/valid/{fake,real}/   DATA_DIR/test/{fake,real}/

Behaviour
  * Up to --epochs epochs (default 20). After every epoch: train and validation loss, accuracy,
    precision, recall, F1, ROC-AUC, AP. Whenever the monitored metric improves (default validation
    ROC-AUC) the weights are saved as the new BEST, replacing the previous best. Training stops after
    --patience epochs without improvement.
  * The validation folder is split deterministically: 60% drives early stopping ('es'), 40% is
    reserved for calibration ('calib', used later by training/calibrate.py). They never overlap.
  * Augmentation: random resized crop, flip, JPEG compression, blur, colour jitter. Label smoothing.
    AdamW + cosine schedule with warmup. Mixed precision (fp16 with GradScaler on the RTX 3050).
  * A full resume checkpoint is written every epoch (--resume continues an interrupted run).
  * test/ is only evaluated when --final-test is given, once, with the best model.

Example
    python training/train_image_model.py --data-dir D:/archive --epochs 20 --patience 4
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from tqdm import tqdm  # noqa: E402

from agents.image_agent.tools.dl_classifier import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402
from eval.datasets import IMAGE_EXTENSIONS, image_subset  # noqa: E402
from eval.metrics import best_threshold, binary_metrics, nan_to_none, train_metrics_row  # noqa: E402
from eval.plots import generate_report_plots, plot_training_curves  # noqa: E402
from training.common import (EarlyStopping, History, atomic_torch_save, format_row, resolve_precision,  # noqa: E402
                             set_seed, stratified_indices, write_json)


class RandomJPEG:
    """Re-encode as JPEG with a random quality (robustness to compression, the main real-world confounder)."""

    def __init__(self, p: float = 0.5, quality=(40, 95)):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p: float = 0.15, radius=(0.1, 1.5)):
        self.p, self.radius = p, radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))


class RandomDownscale:
    """Shrink then re-enlarge: simulates low-resolution / re-sampled sources so the model cannot lean on fine pixel noise."""

    def __init__(self, p: float = 0.3, scale=(0.4, 0.9)):
        self.p, self.scale = p, scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        f = random.uniform(*self.scale)
        small = img.resize((max(16, int(w * f)), max(16, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)


class SelfBlend:
    """Self-blended image (SBI) pseudo-fake: blend a slightly altered copy of a REAL face back onto itself.

    Teaches the blending trace (colour / sharpness / edge mismatch) that all face-swap methods leave, instead of one
    manipulation method's fingerprint -> better zero-shot generalisation to unseen methods.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageChops, ImageEnhance
        img = img.convert("RGB")
        w, h = img.size
        fg = img.copy()
        # 1. random alteration of the "foreground" copy: colour, sharpness, scale, shift
        fg = ImageEnhance.Brightness(fg).enhance(random.uniform(0.85, 1.15))
        fg = ImageEnhance.Contrast(fg).enhance(random.uniform(0.85, 1.15))
        fg = ImageEnhance.Color(fg).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            fg = fg.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        else:
            fg = ImageEnhance.Sharpness(fg).enhance(random.uniform(1.5, 3.0))
        scale = random.uniform(0.94, 1.06)
        fg = fg.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h))
        canvas.paste(img)
        canvas.paste(fg, (int((w - fg.size[0]) / 2) + random.randint(-w // 40, w // 40),
                          int((h - fg.size[1]) / 2) + random.randint(-h // 40, h // 40)))
        fg = canvas
        # 2. soft elliptical face mask (random size / jitter / feathering)
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        rx, ry = random.uniform(0.28, 0.42) * w, random.uniform(0.30, 0.46) * h
        cx, cy = w / 2 + random.uniform(-0.04, 0.04) * w, h / 2 + random.uniform(-0.04, 0.04) * h
        ImageDraw.Draw(mask).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(random.uniform(0.04, 0.12) * w))
        return Image.composite(fg, img, mask)


class SBIFolder(Dataset):
    """Wraps an ImageFolder: with probability p a REAL training image becomes a self-blended pseudo-fake (label fake)."""

    def __init__(self, base, p: float, real_idx: int, fake_idx: int, transform):
        self.base, self.p, self.real, self.fake, self.transform = base, p, real_idx, fake_idx, transform
        self.samples, self.class_to_idx, self.sbi = base.samples, base.class_to_idx, SelfBlend()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        path, label = self.base.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if label == self.real and random.random() < self.p:
            return self.transform(self.sbi(img)), self.fake
        return self.transform(img), label


def transforms_for(train: bool):
    normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + normalize)
    return transforms.Compose([
        RandomDownscale(), RandomJPEG(p=0.6), RandomBlur(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.RandomGrayscale(p=0.05),
    ] + normalize + [transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random")])


class SubsetFolder(Dataset):
    """ImageFolder-style dataset restricted to a list of files (used for the validation early-stop split)."""

    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples, self.transform = samples, transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def es_samples(folder: Path, class_to_idx: dict) -> list[tuple[str, int]]:
    out = []
    for cls, idx in class_to_idx.items():
        for p in sorted((folder / cls).rglob("*")):
            if p.suffix.lower() in IMAGE_EXTENSIONS and image_subset(p.relative_to(folder).as_posix()) == "es":
                out.append((str(p), idx))
    return out


def build_loaders(args, device):
    train_set = datasets.ImageFolder(args.data_dir / "train", transform=transforms_for(True))
    if args.sbi_prob > 0:
        c2i = train_set.class_to_idx
        train_set = SBIFolder(train_set, args.sbi_prob, c2i["real"], c2i["fake"], transforms_for(True))
        print(f"Self-blended pseudo-fakes: p={args.sbi_prob} of real training images")
    valid_folder = args.data_dir / "valid"
    class_to_idx = train_set.class_to_idx
    if set(class_to_idx) != {"fake", "real"}:
        raise ValueError(f"Expected exactly the classes 'fake' and 'real', found {list(class_to_idx)}")
    val_samples = es_samples(valid_folder, class_to_idx)
    if not val_samples:
        raise FileNotFoundError(f"No validation images found under {valid_folder}")
    if args.max_train_samples:
        keep = stratified_indices([y for _, y in train_set.samples], args.max_train_samples, args.seed)
        train_set = torch.utils.data.Subset(train_set, keep)
    if args.max_val_samples:
        keep = stratified_indices([y for _, y in val_samples], args.max_val_samples, args.seed)
        val_samples = [val_samples[i] for i in keep]
    val_set = SubsetFolder(val_samples, transforms_for(False))

    pin = device.type == "cuda"
    kwargs = dict(num_workers=args.num_workers, pin_memory=pin, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False, **kwargs)
    train_labels = [y for _, y in (train_set.dataset.samples if isinstance(train_set, torch.utils.data.Subset)
                                   else train_set.samples)]
    print(f"Train images: {len(train_set)} | validation (early-stop split): {len(val_set)} | classes: {class_to_idx}")
    return train_loader, val_loader, class_to_idx, train_labels


def run_epoch(model, loader, device, amp_dtype, scaler, criterion, fake_idx, desc=""):
    """Evaluation pass: returns (mean loss, labels with 1 = fake, P(fake))."""
    model.eval()
    total_loss, n, probs, labels = 0.0, 0, [], []
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.no_grad():
        for images, y in bar:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
            loss = criterion(logits.float(), y)
            total_loss += float(loss) * len(y)
            n += len(y)
            probs.extend(torch.softmax(logits.float(), dim=1)[:, fake_idx].detach().cpu().tolist())
            labels.extend((y == fake_idx).long().cpu().tolist())          # 1 = fake (positive class)
            bar.set_postfix(loss=f"{total_loss / n:.4f}")
    return total_loss / n, np.array(labels), np.array(probs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("models/efficientnet_b4.pt"), help="best weights (state_dict)")
    ap.add_argument("--checkpoint", type=Path, default=Path("models/efficientnet_b4_last.pt"), help="resume checkpoint")
    ap.add_argument("--results-dir", type=Path, default=Path("eval/results/image_dl"))
    ap.add_argument("--plots-dir", type=Path, default=Path("eval/plots/image_dl"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--monitor", default="val_roc_auc",
                    help="val_roc_auc | val_f1 | val_accuracy | val_average_precision | val_loss")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--class-weights", choices=["auto", "none"], default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-train-samples", type=int, default=None, help="stratified subsample to bound epoch time")
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sbi-prob", type=float, default=0.0,
                    help="probability a real training image is turned into a self-blended pseudo-fake (generalisation)")
    ap.add_argument("--init-weights", type=Path, default=None, help="start from an existing state_dict")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--final-test", action="store_true", help="evaluate the best model on test/ once, after training")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = resolve_precision(args.precision, device)
    print(f"Device: {device} | autocast: {amp_dtype} | grad scaler: {use_scaler}")
    if device.type == "cpu":
        print("WARNING: no GPU detected; training will be very slow.")

    train_loader, val_loader, class_to_idx, train_labels = build_loaders(args, device)
    fake_idx = class_to_idx["fake"]

    model = build_model(pretrained=args.init_weights is None and not args.resume).to(device)
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device, weights_only=True))
        print(f"Initialised from {args.init_weights}")

    if args.class_weights == "auto":
        counts = np.bincount(train_labels, minlength=2).astype(float)
        weights = torch.tensor(len(train_labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss(weight=weights)          # unsmoothed, so train/val losses are comparable

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    stopper = EarlyStopping(args.monitor, args.patience, args.min_delta)
    history = History()
    start_epoch = 1

    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if ckpt["class_to_idx"] != class_to_idx:
            raise ValueError(f"Class mapping mismatch: checkpoint {ckpt['class_to_idx']} vs data {class_to_idx}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        stopper.load_state_dict(ckpt["stopper"])
        history = History(ckpt["history"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed after epoch {ckpt['epoch']} (best {args.monitor}={stopper.best} at epoch {stopper.best_epoch}).")

    meta_path = args.output.with_suffix(".meta.json")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = _train_epoch(model, train_loader, device, amp_dtype, scaler, criterion, eval_criterion,
                                           fake_idx, optimizer, scheduler, f"epoch {epoch} [train]")
        va_loss, va_y, va_p = run_epoch(model, val_loader, device, amp_dtype, scaler, eval_criterion, fake_idx,
                                        desc=f"epoch {epoch} [valid]")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 1)}
        row.update(train_metrics_row("train", tr_y, tr_p, tr_loss))
        row.update(train_metrics_row("val", va_y, va_p, va_loss))

        improved = stopper.update(row[args.monitor], epoch)
        row["is_best"] = improved
        history.append(row)
        print(f"Epoch {epoch:>2}/{args.epochs} | {format_row(row)} | {row['epoch_time_s']}s"
              + ("  <- new best, weights saved" if improved else f"  (no improvement {stopper.num_bad}/{args.patience})"))

        if improved:
            atomic_torch_save(model.state_dict(), args.output)
            write_json({"class_to_idx": class_to_idx, "image_size": IMG_SIZE, "architecture": "efficientnet_b4",
                        "best_epoch": epoch, "monitor": args.monitor, "value": stopper.best,
                        "metrics": row}, meta_path)
            write_json(class_to_idx, args.output.parent / "class_to_idx.json")
        atomic_torch_save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "stopper": stopper.state_dict(),
            "history": history.rows, "class_to_idx": class_to_idx, "args": {k: str(v) for k, v in vars(args).items()},
        }, args.checkpoint)
        history.save(args.results_dir / "history.json", args.results_dir / "history.csv")
        plot_training_curves(history.columns(), args.plots_dir, stopper.best_epoch, "EfficientNet-B4 image classifier")
        if stopper.should_stop:
            print(f"Early stopping: no improvement in {args.monitor} for {args.patience} epochs.")
            break

    if stopper.best is None:
        raise SystemExit("No epoch completed; nothing to report.")
    print(f"\nBest epoch: {stopper.best_epoch} ({args.monitor}={stopper.best:.4f}). Weights: {args.output}")

    best = build_model().to(device)
    best.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    _, va_y, va_p = run_epoch(best, val_loader, device, amp_dtype, scaler, eval_criterion, fake_idx, desc="best model [valid]")
    threshold = best_threshold(va_y, va_p, "f1")
    val_metrics = generate_report_plots(va_y, va_p, args.plots_dir / "validation", "EfficientNet-B4 (validation)", 0.5)
    write_json(nan_to_none({"at_0.5": val_metrics, "f1_optimal_threshold": threshold,
                            "at_f1_optimal": binary_metrics(va_y, va_p, threshold)}), args.results_dir / "val_metrics.json")

    if args.final_test:
        test_dir = args.data_dir / "test"
        if not test_dir.exists():
            raise SystemExit(f"--final-test given but {test_dir} does not exist")
        test_set = datasets.ImageFolder(test_dir, transform=transforms_for(False))
        if test_set.class_to_idx != class_to_idx:
            raise ValueError("Test class mapping differs from training.")
        test_loader = DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
        _, te_y, te_p = run_epoch(best, test_loader, device, amp_dtype, scaler, eval_criterion, fake_idx, desc="test")
        test_metrics = generate_report_plots(te_y, te_p, args.plots_dir / "test", "EfficientNet-B4 (test)", threshold)
        write_json(nan_to_none(test_metrics), args.results_dir / "test_metrics.json")
        print(f"TEST (threshold {threshold:.3f}): acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
              f"auc={test_metrics['roc_auc']:.4f}")
    print(f"Results: {args.results_dir} | Plots: {args.plots_dir}")


def _train_epoch(model, loader, device, amp_dtype, scaler, criterion, eval_criterion, fake_idx, optimizer, scheduler, desc):
    """One training epoch. Optimises `criterion` (label-smoothed) but reports the unsmoothed loss."""
    model.train(True)
    total, n, probs, labels = 0.0, 0, [], []
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, y in bar:
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
        loss = criterion(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        with torch.no_grad():
            total += float(eval_criterion(logits.float(), y)) * len(y)
        n += len(y)
        probs.extend(torch.softmax(logits.float(), dim=1)[:, fake_idx].detach().cpu().tolist())
        labels.extend((y == fake_idx).long().cpu().tolist())
        bar.set_postfix(loss=f"{total / n:.4f}")
    return total / n, np.array(labels), np.array(probs)


if __name__ == "__main__":
    main()
