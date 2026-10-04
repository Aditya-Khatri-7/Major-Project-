"""Shared training utilities: seeding, early stopping, epoch history, mixed-precision selection, atomic saves."""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np

MIN_MODES = {"val_loss", "train_loss"}


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def monitor_mode(monitor: str) -> str:
    """'min' for losses, 'max' for everything else (AUC, F1, accuracy, ...)."""
    return "min" if monitor in MIN_MODES else "max"


class EarlyStopping:
    """Tracks the best value of a monitored metric.

    `update()` returns True when the epoch is a new best (the caller then saves the model, replacing
    the previous best checkpoint). Training stops after `patience` epochs without an improvement of
    at least `min_delta`.
    """

    def __init__(self, monitor: str = "val_roc_auc", patience: int = 4, min_delta: float = 1e-4):
        self.monitor = monitor
        self.mode = monitor_mode(monitor)
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.num_bad = 0

    def update(self, value: float, epoch: int) -> bool:
        if value != value:                                  # NaN never counts as an improvement
            self.num_bad += 1
            return False
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best + self.min_delta)
            or (self.mode == "min" and value < self.best - self.min_delta)
        )
        if improved:
            self.best, self.best_epoch, self.num_bad = value, epoch, 0
        else:
            self.num_bad += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.num_bad >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "num_bad": self.num_bad}

    def load_state_dict(self, state: dict) -> None:
        self.best, self.best_epoch, self.num_bad = state["best"], state["best_epoch"], state["num_bad"]


class History:
    """Per-epoch metric log, stored column-wise (JSON) and row-wise (CSV)."""

    def __init__(self, rows: Optional[list[dict]] = None):
        self.rows: list[dict] = rows or []

    def append(self, row: dict) -> None:
        self.rows.append(row)

    def columns(self) -> dict[str, list]:
        keys: list[str] = []
        for r in self.rows:
            keys.extend(k for k in r if k not in keys)
        return {k: [r.get(k) for r in self.rows] for k in keys}

    def save(self, json_path: str | Path, csv_path: Optional[str | Path] = None) -> None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.columns(), indent=2), encoding="utf-8")
        if csv_path:
            cols = self.columns()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(cols))
                writer.writeheader()
                writer.writerows(self.rows)


def resolve_precision(precision: str, device) -> tuple[Optional["object"], bool]:
    """Return (autocast dtype or None, use GradScaler). 'auto' prefers bf16, falls back to fp16."""
    import torch

    if device.type != "cuda" or precision == "fp32":
        return None, False
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, True


def atomic_torch_save(obj, path: str | Path) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written checkpoint."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def stratified_indices(labels: list[int], n: int, seed: int) -> list[int]:
    """Random subset of `n` indices with the class ratio preserved."""
    if n >= len(labels):
        return list(range(len(labels)))
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    chosen: list[int] = []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        chosen.extend(idxs[: max(1, round(n * len(idxs) / len(labels)))])
    rng.shuffle(chosen)
    return chosen[:n]


def format_row(row: dict) -> str:
    keys = ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_precision", "val_recall",
            "val_f1", "val_roc_auc")
    return " | ".join(f"{k.replace('_accuracy', '_acc').replace('_roc_auc', '_auc')}={row[k]:.4f}" for k in keys if k in row)
