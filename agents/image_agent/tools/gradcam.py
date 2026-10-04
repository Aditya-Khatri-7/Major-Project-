"""Grad-CAM localisation heatmap for the EfficientNet classifier (explainability output)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def generate_gradcam(model, input_tensor, target_index: int, target_layer=None) -> np.ndarray:
    """Return a [H, W] heatmap in [0, 1] showing where the model looked for `target_index`.

    `input_tensor` is a [1, 3, H, W] float32 tensor on the model's device. The model must be in
    eval mode and run in fp32. Tensor hooks are used (not module backward hooks) so in-place
    activations such as SiLU(inplace=True) do not break the backward pass.
    """
    import torch
    import torch.nn.functional as F

    layer = target_layer if target_layer is not None else model.features[-1]
    store: dict = {}

    def forward_hook(_module, _inputs, output):
        store["activations"] = output
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = input_tensor.detach().clone().requires_grad_(True)
            logits = model(x)
            logits[0, target_index].backward()
        activations = store["activations"].detach()
        gradients = store["gradients"].detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def save_overlay(image, heatmap: np.ndarray, out_path: str | Path, alpha: float = 0.45) -> Path:
    """Blend the heatmap over the (PIL) image and save it as a PNG."""
    import matplotlib
    from PIL import Image

    h, w = heatmap.shape
    base = np.asarray(image.convert("RGB").resize((w, h)), dtype=np.float32) / 255.0
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3].astype(np.float32)
    blended = np.clip((1 - alpha) * base + alpha * colored, 0.0, 1.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((blended * 255).astype(np.uint8)).save(out_path)
    return out_path


def try_generate(model, input_tensor, target_index: int, image, out_path: Optional[str | Path]) -> Optional[str]:
    """Generate and save the overlay; return its path, or None if anything goes wrong."""
    if out_path is None:
        return None
    try:
        heatmap = generate_gradcam(model, input_tensor, target_index)
        return str(save_overlay(image, heatmap, out_path))
    except Exception:
        return None
