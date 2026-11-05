"""Whole-image detector for modern AI generators (SwinV2, haywoodsloan/ai-image-detector-deploy, Apache-2.0).

The face tools were trained on older face swaps/edits and tight face crops; they scored a modern AI-generated bedroom selfie as authentic.
This tool looks at the whole image and runs on every upload. On OpenFake-tiny test (Flux.2, Midjourney 7, GPT-image-1.5 ...) it reaches
AUROC 0.91 where the old general probe scored 0.38. Weights live in models/modern_detector (git-ignored; see docs/MODELS.md).
"""
from __future__ import annotations

import threading
from pathlib import Path

from agents.calibration import load_calibration, tool_confidence, tool_probability
from agents.schemas import ToolVerdict

TOOL_NAME = "image_modern"
MODEL_DIR = Path("models/modern_detector")
_lock = threading.Lock()
_loaded: dict = {}


def _load() -> dict:
    with _lock:
        if not _loaded:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            if not (MODEL_DIR / "model.safetensors").exists():
                raise FileNotFoundError(f"Modern-generator detector not found at '{MODEL_DIR}'. Download haywoodsloan/ai-image-detector-deploy.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, use_safetensors=True).to(device, dtype).eval()
            labels = {int(i): str(name).lower() for i, name in model.config.id2label.items()}
            fake = next(i for i, name in labels.items() if name in ("artificial", "ai", "fake"))
            real = next(i for i in labels if i != fake)
            _loaded.update(model=model, proc=AutoImageProcessor.from_pretrained(MODEL_DIR), device=device, dtype=dtype, fake=fake, real=real)
        return _loaded


def score_image(image_path: str, **_) -> ToolVerdict:
    import torch
    from PIL import Image, ImageOps

    try:
        ctx = _load()
    except FileNotFoundError as exc:
        return ToolVerdict.failure(TOOL_NAME, "image", str(exc))
    with Image.open(image_path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    x = ctx["proc"](images=img, return_tensors="pt")["pixel_values"].to(ctx["device"], ctx["dtype"])
    with torch.no_grad():
        logits = ctx["model"](pixel_values=x).logits.float()[0]
    raw = {"raw_logit": float(logits[ctx["fake"]] - logits[ctx["real"]]), "conf_factor": 1.0}
    p = tool_probability(TOOL_NAME, raw, load_calibration())
    return ToolVerdict(tool_name=TOOL_NAME, modality="image", score=p, confidence=tool_confidence(p, raw),
                       explanation=f"Whole-image modern-generator detector (SwinV2) P(AI-generated)={p:.3f}.", raw_features=raw)
