"""Fine-tune DeBERTa-v3-small to separate human from AI-generated text.

Data: CSVs produced by `python data/prepare_text.py hc3` (columns: id, text, label 0=human 1=AI).

Behaviour
  * Trains for up to --epochs epochs. After every epoch it computes train and validation metrics
    (loss, accuracy, precision, recall, F1, ROC-AUC, AP). Whenever the monitored metric improves
    (default: validation ROC-AUC) the model is saved as the new BEST, replacing the previous best.
    Training stops early after --patience epochs without improvement.
  * A full resume checkpoint is written every epoch (--resume continues an interrupted run).
  * Curves, ROC/PR/confusion/calibration plots and metrics JSON/CSV are written to eval/.
  * The test CSV is only evaluated when --final-test is given, once, with the best model.

Example
    python training/train_text_classifier.py --train-csv data/processed/text/hc3_train.csv \
        --val-csv data/processed/text/hc3_val.csv --epochs 20 --patience 3
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from eval.datasets import load_text_rows  # noqa: E402
from eval.metrics import best_threshold, binary_metrics, nan_to_none, train_metrics_row  # noqa: E402
from eval.plots import generate_report_plots, plot_training_curves  # noqa: E402
from training.common import (EarlyStopping, History, atomic_torch_save, format_row, resolve_precision,  # noqa: E402
                             set_seed, write_json)

LABELS = {"human": 0, "ai": 1}


class TextDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, crop_words: int, seed: int):
        self.rows, self.train, self.crop_words = rows, train, crop_words
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[str, int]:
        row = self.rows[i]
        text = row["text"]
        if self.train:                                   # random word-span crop: matches window inference
            words = text.split()
            if len(words) > self.crop_words:
                start = self.rng.randint(0, len(words) - self.crop_words)
                text = " ".join(words[start:start + self.crop_words])
        return text, row["label"]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


def run_epoch(model, loader, device, amp_dtype, scaler, criterion, optimizer=None, scheduler=None,
              accum_steps: int = 1, desc: str = "", max_grad_norm: float = 1.0):
    train = optimizer is not None
    model.train(train)
    losses, weights, probs, labels = 0.0, 0, [], []
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    if train:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for step, batch in enumerate(bar):
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**batch).logits
            loss = criterion(logits.float(), y)
            if train:
                scaler.scale(loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            losses += float(loss) * len(y)
            weights += len(y)
            probs.extend(torch.softmax(logits.float(), dim=1)[:, LABELS["ai"]].detach().cpu().tolist())
            labels.extend(y.cpu().tolist())
            bar.set_postfix(loss=f"{losses / weights:.4f}")
    return losses / weights, np.array(labels), np.array(probs)


def load_base_model(name: str):
    """Load the pretrained encoder with a 2-class head.

    Some hub models (e.g. microsoft/deberta-v3-small) only ship a pickle checkpoint on `main`, which recent
    transformers refuse to load with torch < 2.6. The hub's auto-converted safetensors copy lives on
    `refs/pr/4`; fall back to it, or upgrade torch to >= 2.6.
    """
    from transformers import AutoModelForSequenceClassification

    kwargs = dict(dtype=torch.float32, num_labels=2, id2label={0: "human", 1: "ai"}, label2id=LABELS)
    try:
        return AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    except (ValueError, OSError) as exc:
        print(f"Default weights could not be loaded ({str(exc)[:120]}...). Retrying with safetensors from refs/pr/4.")
        return AutoModelForSequenceClassification.from_pretrained(name, revision="refs/pr/4", use_safetensors=True, **kwargs)


def save_best(model, tokenizer, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_json(meta, out_dir / "train_meta.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--test-csv", type=Path, default=None)
    ap.add_argument("--final-test", action="store_true", help="evaluate the best model on --test-csv once, after training")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-small")
    ap.add_argument("--output-dir", type=Path, default=Path("models/text_dl"), help="best model (HF format)")
    ap.add_argument("--checkpoint", type=Path, default=Path("models/text_dl_last.pt"), help="resume checkpoint")
    ap.add_argument("--results-dir", type=Path, default=Path("eval/results/text_dl"))
    ap.add_argument("--plots-dir", type=Path, default=Path("eval/plots/text_dl"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--monitor", default="val_roc_auc",
                    help="val_roc_auc | val_f1 | val_accuracy | val_average_precision | val_loss")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--crop-words", type=int, default=200, help="random training crop, in words")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--class-weights", choices=["auto", "none"], default="auto")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = resolve_precision(args.precision, device)
    print(f"Device: {device} | autocast: {amp_dtype} | grad scaler: {use_scaler}")
    if device.type == "cpu":
        print("WARNING: no GPU detected; training will be very slow.")

    train_rows = load_text_rows(args.train_csv, args.max_train_samples, args.seed)
    val_rows = load_text_rows(args.val_csv, args.max_val_samples, args.seed)
    print(f"Train: {len(train_rows)} rows ({np.mean([r['label'] for r in train_rows]):.1%} AI) | "
          f"Validation: {len(val_rows)} rows")

    from agents.text_agent.tools.dl_classifier import load_tokenizer

    tokenizer = load_tokenizer(args.model_name)
    model = load_base_model(args.model_name).to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(TextDataset(train_rows, True, args.crop_words, args.seed), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows, False, args.crop_words, args.seed), batch_size=args.batch_size * 2,
                            shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    if args.class_weights == "auto":
        n_ai = sum(r["label"] for r in train_rows)
        n_h = len(train_rows) - n_ai
        weights = torch.tensor([len(train_rows) / (2 * max(n_h, 1)), len(train_rows) / (2 * max(n_ai, 1))],
                               dtype=torch.float32, device=device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    stopper = EarlyStopping(args.monitor, args.patience, args.min_delta)
    history = History()
    start_epoch = 1
    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        stopper.load_state_dict(ckpt["stopper"])
        history = History(ckpt["history"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed after epoch {ckpt['epoch']} (best {args.monitor}={stopper.best} at epoch {stopper.best_epoch}).")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, device, amp_dtype, scaler, criterion, optimizer, scheduler,
                                        args.grad_accum, f"epoch {epoch} [train]")
        va_loss, va_y, va_p = run_epoch(model, val_loader, device, amp_dtype, scaler, criterion, desc=f"epoch {epoch} [valid]")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 1)}
        row.update(train_metrics_row("train", tr_y, tr_p, tr_loss))
        row.update(train_metrics_row("val", va_y, va_p, va_loss))

        improved = stopper.update(row[args.monitor], epoch)
        row["is_best"] = improved
        history.append(row)
        print(f"Epoch {epoch:>2}/{args.epochs} | {format_row(row)} | {row['epoch_time_s']}s"
              + ("  <- new best, model saved" if improved else f"  (no improvement {stopper.num_bad}/{args.patience})"))

        if improved:
            save_best(model, tokenizer, args.output_dir, {
                "epoch": epoch, "monitor": args.monitor, "value": stopper.best, "base_model": args.model_name,
                "max_length": args.max_length, "labels": LABELS, "metrics": {k: v for k, v in row.items()},
            })
        atomic_torch_save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "stopper": stopper.state_dict(), "history": history.rows, "args": vars(args),
        }, args.checkpoint)
        history.save(args.results_dir / "history.json", args.results_dir / "history.csv")
        plot_training_curves(history.columns(), args.plots_dir, stopper.best_epoch, "DeBERTa text classifier")
        if stopper.should_stop:
            print(f"Early stopping: no improvement in {args.monitor} for {args.patience} epochs.")
            break

    if stopper.best is None:
        raise SystemExit("No epoch completed; nothing to report.")
    print(f"\nBest epoch: {stopper.best_epoch} ({args.monitor}={stopper.best:.4f}). Model: {args.output_dir}")

    # ---- report on the best model (validation) ----
    best = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    _, va_y, va_p = run_epoch(best, val_loader, device, amp_dtype, scaler, criterion, desc="best model [valid]")
    threshold = best_threshold(va_y, va_p, "f1")
    val_metrics = generate_report_plots(va_y, va_p, args.plots_dir / "validation", "DeBERTa (validation)", 0.5)
    write_json(nan_to_none({"at_0.5": val_metrics, "f1_optimal_threshold": threshold,
                            "at_f1_optimal": binary_metrics(va_y, va_p, threshold)}), args.results_dir / "val_metrics.json")

    if args.final_test:
        if not args.test_csv:
            raise SystemExit("--final-test requires --test-csv")
        test_rows = load_text_rows(args.test_csv, None, args.seed)
        test_loader = DataLoader(TextDataset(test_rows, False, args.crop_words, args.seed), batch_size=args.batch_size * 2,
                                 shuffle=False, collate_fn=collate)
        _, te_y, te_p = run_epoch(best, test_loader, device, amp_dtype, scaler, criterion, desc="test")
        test_metrics = generate_report_plots(te_y, te_p, args.plots_dir / "test", "DeBERTa (test)", threshold)
        write_json(nan_to_none(test_metrics), args.results_dir / "test_metrics.json")
        print(f"TEST (threshold {threshold:.3f}): acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
              f"auc={test_metrics['roc_auc']:.4f}")
    print(f"Results: {args.results_dir} | Plots: {args.plots_dir}")


if __name__ == "__main__":
    main()
