"""Image tool for NON-face images: frozen CLIP ViT-L/14 + logistic probe trained on GRAVEX-200K (real vs AI images).

Used when the scope guard says the image is not a face photo (illustration, AI art, landscape...). The face-deepfake
tools are skipped for those images because they were never trained on them. Held-out GRAVEX AUC 0.94; not calibrated.
"""
from __future__ import annotations

import threading
from pathlib import Path

from agents.calibration import load_calibration, tool_confidence, tool_probability
from agents.schemas import ToolVerdict

TOOL_NAME = "image_general"
PROBE_PATH = Path("models/clip_general.joblib")
_lock = threading.Lock()
_loaded: dict = {}


def _clf():
    with _lock:
        if "clf" not in _loaded:
            import joblib
            if not PROBE_PATH.exists():
                raise FileNotFoundError(f"General AI-image probe not found at '{PROBE_PATH}'. Run: python eval/clip_general.py")
            _loaded["clf"] = joblib.load(PROBE_PATH)["clf"]
        return _loaded["clf"]


def score_image(image_path: str, **_) -> ToolVerdict:
    import torch
    from PIL import Image

    from agents.image_agent.tools.clip_probe import _load

    try:
        clf, ctx = _clf(), _load()
    except FileNotFoundError as exc:
        return ToolVerdict.failure(TOOL_NAME, "image", str(exc))
    with Image.open(image_path) as im:
        x = ctx["proc"](images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(ctx["device"], ctx["dtype"])
    with torch.no_grad():
        f = ctx["model"].get_image_features(pixel_values=x)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
    raw = {"raw_logit": float(clf.decision_function(f)[0]), "conf_factor": 1.0}
    p = tool_probability(TOOL_NAME, raw, load_calibration())
    return ToolVerdict(tool_name=TOOL_NAME, modality="image", score=p, confidence=tool_confidence(p, raw),
                       explanation=f"General AI-image probe (CLIP, non-face images) P(AI-generated)={p:.3f}.", raw_features=raw)
