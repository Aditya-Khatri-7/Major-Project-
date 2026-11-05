"""Pixel-level AI-edit localiser (DeepLabV3-ResNet50 trained on real Stable Diffusion inpainting edits with exact masks).

Output is a probability map P(pixel was AI-edited/synthesised). Image-level score = high percentile of the map.
Validated (held-out photos, edits from the same tool): AUROC 0.98 at 2% edit size, 1.0 from 4%;
clean unedited images flagged 0.7%, 9.5% after JPEG q75.
Cross-pipeline: DeepFakeFace inpainting 97% flagged, text-to-image 91%, real 0.7%; it does NOT detect face swaps (insight 9%, Celeb-DF 15%).
Not validated on other editors (Photoshop generative fill, Flux/SDXL/GPT-image edits, retouching apps).
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np

from agents.calibration import load_calibration, tool_confidence, tool_probability
from agents.schemas import ToolVerdict

TOOL_NAME = "image_edit"
MODEL_PATH = Path("models/localizer_v1.pt")
SIZE = 512
PERCENTILE = 99.5
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
_lock = threading.Lock()
_loaded: dict = {}


def _load() -> dict:
    with _lock:
        if not _loaded:
            import torch

            from training.train_localizer import build
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Edit localiser weights not found at '{MODEL_PATH}'. Run: python training/train_localizer.py")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build().to(device).eval()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            _loaded.update(model=model, device=device)
        return _loaded


def prob_map(img) -> np.ndarray:
    """Per-pixel edit probability at SIZE x SIZE for a PIL image."""
    import torch
    ctx = _load()
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE)), np.float32) / 255.0
    x = torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1))[None].to(ctx["device"])
    with torch.no_grad(), torch.autocast(ctx["device"], dtype=torch.bfloat16, enabled=ctx["device"] == "cuda"):
        return torch.sigmoid(ctx["model"](x)["out"].float())[0, 0].cpu().numpy()


def score_image(image_path: str, **_) -> ToolVerdict:
    from PIL import Image, ImageOps

    try:
        _load()
    except FileNotFoundError as exc:
        return ToolVerdict.failure(TOOL_NAME, "image", str(exc))
    with Image.open(image_path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    m = prob_map(img)
    p995 = float(np.percentile(m, PERCENTILE))
    p995 = min(max(p995, 1e-4), 1 - 1e-4)
    raw = {"raw_logit": math.log(p995 / (1 - p995)), "conf_factor": 1.0, "edited_area": float((m >= 0.5).mean()), "map_max": float(m.max())}
    p = tool_probability(TOOL_NAME, raw, load_calibration())
    return ToolVerdict(tool_name=TOOL_NAME, modality="image", score=p, confidence=tool_confidence(p, raw),
                       explanation=f"Pixel-level AI-edit localiser: {raw['edited_area'] * 100:.1f}% of pixels flagged, image score P={p:.3f}.",
                       raw_features=raw)
