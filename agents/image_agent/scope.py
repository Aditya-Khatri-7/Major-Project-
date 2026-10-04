"""Scope guard: is this a photo/crop of a human face? (the domain our deepfake detectors were trained on).

Zero-shot CLIP: compares the image with 'a face' prompts against 'not a face / artwork' prompts. Out-of-scope images
(illustrations, landscapes, logos, AI art without a real face) get no confident verdict from the face-deepfake tools.
"""
from __future__ import annotations

FACE_PROMPTS = ["a close-up photograph of a person's face", "a photo of a human face", "a portrait photo of a person",
                "a cropped face from a video frame"]
OTHER_PROMPTS = ["an illustration or digital artwork", "a cartoon or painting", "a landscape or object photograph",
                 "a logo, diagram or text graphic", "an abstract graphic design", "a photo with no people in it"]
THRESHOLD = 0.5


def face_probability(image_path: str) -> float:
    import torch
    from PIL import Image

    from agents.image_agent.tools.clip_probe import _load

    ctx = _load()
    model, proc, device = ctx["model"], ctx["proc"], ctx["device"]
    from transformers import CLIPTokenizer
    if "scope_text" not in ctx:
        tok = CLIPTokenizer.from_pretrained(model.config._name_or_path)
        t = tok(FACE_PROMPTS + OTHER_PROMPTS, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            te = model.get_text_features(**t)
            te = te.pooler_output if hasattr(te, "pooler_output") else te
        ctx["scope_text"] = torch.nn.functional.normalize(te.float(), dim=-1)
    with Image.open(image_path) as im:
        x = proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(device, ctx["dtype"])
    with torch.no_grad():
        f = model.get_image_features(pixel_values=x)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        probs = (100.0 * f @ ctx["scope_text"].T).softmax(dim=-1)[0]
    return float(probs[: len(FACE_PROMPTS)].sum())
