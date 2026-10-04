"""Image tool: frozen CLIP ViT-L/14 embedding + logistic-regression probe (fitted by eval/clip_probe.py).

Complements the fine-tuned EfficientNet: the CNN learns manipulation-specific pixel fingerprints, the CLIP probe sees
semantic / global statistics. Their errors are partly independent (unseen-manipulation AUC 0.755 -> 0.798 when averaged),
so the verifier can use both. Weaker alone on seen methods (FF++ seen 0.83 vs 0.98), so fusion weights reflect that.
"""
from __future__ import annotations

import threading
from pathlib import Path

from agents.calibration import load_calibration, tool_confidence, tool_probability
from agents.schemas import ToolVerdict

TOOL_NAME = "image_clip"
PROBE_PATH = Path("models/clip_probe_v2.joblib")   # unified face-swap + AI-synthesised-face probe (eval/clip_face_unified.py)

_lock = threading.Lock()
_loaded: dict = {}


def _load() -> dict:
    with _lock:
        if _loaded:
            return _loaded
        import joblib
        import torch
        from transformers import CLIPImageProcessor, CLIPModel

        if not PROBE_PATH.exists():
            raise FileNotFoundError(f"CLIP probe not found at '{PROBE_PATH}'. Run: python eval/clip_probe.py extract && fit")
        bundle = joblib.load(PROBE_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = CLIPModel.from_pretrained(bundle["model"], dtype=dtype).to(device).eval()
        _loaded.update(clf=bundle["clf"], proc=CLIPImageProcessor.from_pretrained(bundle["model"]),
                       model=model, device=device, dtype=dtype)
        return _loaded


def score_image(image_path: str, **_) -> ToolVerdict:
    import torch
    from PIL import Image

    try:
        ctx = _load()
    except FileNotFoundError as exc:
        return ToolVerdict.failure(TOOL_NAME, "image", str(exc))
    with Image.open(image_path) as im:
        x = ctx["proc"](images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(ctx["device"], ctx["dtype"])
    with torch.no_grad():
        f = ctx["model"].get_image_features(pixel_values=x)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
    raw_logit = float(ctx["clf"].decision_function(f)[0])
    raw = {"raw_logit": raw_logit, "conf_factor": 1.0}
    p = tool_probability(TOOL_NAME, raw, load_calibration())
    return ToolVerdict(tool_name=TOOL_NAME, modality="image", score=p, confidence=tool_confidence(p, raw),
                       explanation=f"CLIP ViT-L/14 linear probe P(fake)={p:.3f}.", raw_features=raw)
