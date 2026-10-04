"""Text DL tool: fine-tuned DeBERTa-v3 sequence classifier (trained by training/train_text_classifier.py).

Long inputs are split into overlapping token windows and the window logits are averaged.
The averaged logit is converted to a calibrated probability with temperature scaling.
"""
from __future__ import annotations

import statistics
import threading
from pathlib import Path

from agents.calibration import load_calibration, sigmoid, tool_confidence, tool_probability
from agents.schemas import ToolVerdict
from agents.text_agent.normalize import normalize_text
from config import settings

TOOL_NAME = "text_dl"
WINDOW_TOKENS = 254          # content tokens per window (+2 special tokens = 256)
STRIDE_TOKENS = 127
MAX_WINDOWS = 8
MAX_WORDS = 6000

_lock = threading.Lock()
_loaded: dict = {}


def load_tokenizer(path_or_name):
    """Slow (sentencepiece) tokenizer, used for BOTH training and inference so they tokenize identically.

    The fast DeBERTa-v3 tokenizer written by save_pretrained() cannot be reloaded with some transformers versions.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path_or_name, use_fast=False)


def _load() -> dict:
    with _lock:
        if _loaded:
            return _loaded
        import torch
        from transformers import AutoModelForSequenceClassification

        path = Path(settings.text_dl_model_path)
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"Text DL model not found at '{path}'. Train it first (see TRAINING_GUIDE.md, step 4)."
            )
        tokenizer = load_tokenizer(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        ai_index = int(model.config.label2id.get("ai", 1))
        _loaded.update(tokenizer=tokenizer, model=model, device=device, ai_index=ai_index, human_index=1 - ai_index)
        return _loaded


def _windows(tokenizer, text: str) -> list[list[int]]:
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        return []
    windows, start = [], 0
    while True:
        windows.append(ids[start:start + WINDOW_TOKENS])
        if start + WINDOW_TOKENS >= len(ids) or len(windows) >= MAX_WINDOWS:
            break
        start += STRIDE_TOKENS
    return windows


def window_logits(text: str) -> list[float]:
    """Raw (AI minus human) logit for each window of the text."""
    import torch

    ctx = _load()
    tokenizer, model, device = ctx["tokenizer"], ctx["model"], ctx["device"]
    text, _ = normalize_text(text)           # undo homoglyph / zero-width attacks before tokenizing
    windows = _windows(tokenizer, text)
    if not windows:
        return []
    batch = tokenizer.pad(
        {"input_ids": [[tokenizer.cls_token_id] + w + [tokenizer.sep_token_id] for w in windows]}, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**batch).logits.float()
    diff = logits[:, ctx["ai_index"]] - logits[:, ctx["human_index"]]
    return [float(x) for x in diff.cpu()]


def score_text(text: str) -> ToolVerdict:
    try:
        diffs = window_logits(text)
    except FileNotFoundError as exc:
        return ToolVerdict.failure(TOOL_NAME, "text", str(exc))
    if not diffs:
        return ToolVerdict.failure(TOOL_NAME, "text", "Text produced no tokens.")

    cal = load_calibration()
    raw_logit = sum(diffs) / len(diffs)
    window_probs = [sigmoid(d / max(float(cal["text_dl"]["temperature"]), 1e-3)) for d in diffs]
    spread = statistics.pstdev(window_probs) if len(window_probs) > 1 else 0.0
    _, norm_stats = normalize_text(text)
    raw = {
        **norm_stats,
        "raw_logit": raw_logit, "n_windows": len(diffs),
        "conf_factor": 1.0 - min(spread, 0.5), "window_std": spread,
    }
    p = tool_probability(TOOL_NAME, raw, cal)
    return ToolVerdict(
        tool_name=TOOL_NAME, modality="text", score=p, confidence=tool_confidence(p, raw),
        explanation=f"DeBERTa classifier P(AI)={p:.3f} over {len(diffs)} window(s).",
        raw_features=raw,
    )
