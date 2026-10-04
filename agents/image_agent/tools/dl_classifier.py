"""Image DL tool: fine-tuned EfficientNet-B4 real-vs-fake classifier with Grad-CAM localisation.

Weights are produced by training/train_image_model.py (best epoch by validation AUC).
The raw (fake minus real) logit is converted to a calibrated probability by temperature scaling.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from agents.calibration import load_calibration, tool_confidence, tool_probability
from agents.image_agent.tools.gradcam import try_generate
from agents.schemas import ToolVerdict
from config import settings

TOOL_NAME = "image_dl"
IMG_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LEGACY_WEIGHTS = "efficientnet_b4_wilddeepfake.pt"

_lock = threading.Lock()
_loaded: dict = {}


def _weights_path() -> Path:
    path = Path(settings.image_model_path)
    if not path.exists():
        legacy = path.with_name(LEGACY_WEIGHTS)
        if legacy.exists():
            return legacy
    return path


def _class_index(path: Path) -> dict:
    for candidate in (path.with_suffix(".meta.json"), path.parent / "class_to_idx.json"):
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            mapping = data.get("class_to_idx", data)
            if "fake" in mapping and "real" in mapping:
                return {"fake": int(mapping["fake"]), "real": int(mapping["real"])}
            raise ValueError(f"{candidate} must define both 'fake' and 'real' class indices, got {mapping}")
    return {"fake": 0, "real": 1}       # torchvision ImageFolder default (alphabetical)


def build_model(pretrained: bool = False):
    import torch.nn as nn
    from torchvision import models

    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def eval_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load() -> dict:
    with _lock:
        if _loaded:
            return _loaded
        import torch

        path = _weights_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Image DL weights not found at '{settings.image_model_path}'. "
                "Train the model first (see TRAINING_GUIDE.md, step 6)."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model = build_model()
        model.load_state_dict(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        _loaded.update(model=model, device=device, index=_class_index(path), transform=eval_transform())
        return _loaded


def score_image(image_path: str, with_gradcam: bool = True, gradcam_path: Optional[str] = None) -> ToolVerdict:
    import torch
    from PIL import Image, ImageOps

    try:
        ctx = _load()
    except FileNotFoundError as exc:
        return ToolVerdict.failure(TOOL_NAME, "image", str(exc))

    model, device, index = ctx["model"], ctx["device"], ctx["index"]
    try:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        base = ctx["transform"](image)
        views = [base, torch.flip(base, dims=[2])] if settings.image_tta else [base]
        batch = torch.stack(views).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch).float()
        raw_logit = float((logits[:, index["fake"]] - logits[:, index["real"]]).mean())
    except Exception as exc:
        return ToolVerdict.failure(TOOL_NAME, "image", f"{type(exc).__name__}: {exc}")

    cal = load_calibration()
    raw = {"raw_logit": raw_logit, "conf_factor": 1.0, "tta_views": len(views)}
    p = tool_probability(TOOL_NAME, raw, cal)

    heatmap_path = None
    if with_gradcam:
        target = gradcam_path or str(Path(image_path).with_name("gradcam.png"))
        heatmap_path = try_generate(model, batch[:1].float(), index["fake"], image, target)
        if heatmap_path:
            raw["gradcam_path"] = heatmap_path

    note = " Grad-CAM heatmap saved." if heatmap_path else ""
    return ToolVerdict(
        tool_name=TOOL_NAME, modality="image", score=p, confidence=tool_confidence(p, raw),
        explanation=f"EfficientNet-B4 P(fake)={p:.3f}.{note}", raw_features=raw,
    )
