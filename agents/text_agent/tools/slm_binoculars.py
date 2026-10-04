"""Text SLM tool: zero-shot Binoculars detector with a small observer/performer pair.

Binoculars (arXiv:2401.12070) scores a text as
    B(s) = log-perplexity of s under the performer model
           / cross-perplexity between the observer's and the performer's next-token distributions.
LOW scores indicate machine-generated text. Defaults: Qwen2.5-0.5B (observer) and
Qwen2.5-0.5B-Instruct (performer), about 2 GB in fp16, so it fits a 6 GB GPU.

The raw score is mapped to P(AI) with a logistic fitted on calibration data (training/calibrate.py).
"""
from __future__ import annotations

import threading

from agents.calibration import load_calibration, tool_confidence, tool_probability
from agents.schemas import ToolVerdict
from config import settings

TOOL_NAME = "text_slm"
MIN_WORDS = 50
MIN_TOKENS = 16
MAX_TOKENS = 512
_CHUNK = 128                 # positions processed at a time when computing cross-perplexity

_lock = threading.Lock()
_loaded: dict = {}


def _load() -> dict:
    with _lock:
        if _loaded:
            return _loaded
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(settings.slm_observer_model)
        performer_tok = AutoTokenizer.from_pretrained(settings.slm_performer_model)
        if tokenizer.get_vocab() != performer_tok.get_vocab():
            raise ValueError("Observer and performer models must share a tokenizer/vocabulary.")
        observer = AutoModelForCausalLM.from_pretrained(settings.slm_observer_model, torch_dtype=dtype).to(device).eval()
        performer = AutoModelForCausalLM.from_pretrained(settings.slm_performer_model, torch_dtype=dtype).to(device).eval()
        _loaded.update(tokenizer=tokenizer, observer=observer, performer=performer, device=device)
        return _loaded


def binoculars_score(text: str) -> tuple[float, int]:
    """Return (binoculars score, number of scored tokens)."""
    import torch
    import torch.nn.functional as F

    ctx = _load()
    enc = ctx["tokenizer"](text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(ctx["device"])
    with torch.no_grad():
        obs_logits = ctx["observer"](**enc).logits[:, :-1].float()
        perf_logits = ctx["performer"](**enc).logits[:, :-1].float()
        labels = enc["input_ids"][:, 1:]
        n_tokens = int(labels.shape[1])
        if n_tokens < MIN_TOKENS:
            raise ValueError(f"only {n_tokens} tokens")
        log_ppl = F.cross_entropy(perf_logits.transpose(1, 2), labels, reduction="mean")
        total, count = 0.0, 0
        for i in range(0, n_tokens, _CHUNK):
            p_obs = F.softmax(obs_logits[:, i:i + _CHUNK], dim=-1)
            logp_perf = F.log_softmax(perf_logits[:, i:i + _CHUNK], dim=-1)
            total += float(-(p_obs * logp_perf).sum(dim=-1).sum())
            count += p_obs.shape[1]
        x_ppl = total / count
    return float(log_ppl) / x_ppl, n_tokens


def score_text(text: str) -> ToolVerdict:
    if len(text.split()) < MIN_WORDS:
        return ToolVerdict.failure(TOOL_NAME, "text", f"Text shorter than {MIN_WORDS} words; Binoculars is unreliable.")
    try:
        score, n_tokens = binoculars_score(text)
    except Exception as exc:
        return ToolVerdict.failure(TOOL_NAME, "text", f"{type(exc).__name__}: {exc}")

    cal = load_calibration()
    calibrated = bool(cal["text_slm"].get("calibrated", True))
    raw = {"binoculars_score": score, "n_tokens": n_tokens, "conf_factor": 1.0 if calibrated else 0.5}
    p = tool_probability(TOOL_NAME, raw, cal)
    note = "" if calibrated else " (uncalibrated: run training/calibrate.py)"
    return ToolVerdict(
        tool_name=TOOL_NAME, modality="text", score=p, confidence=tool_confidence(p, raw),
        explanation=f"Binoculars score={score:.3f} (low = machine-like) -> P(AI)={p:.3f}{note}.",
        raw_features={**raw, "calibrated": calibrated},
    )
