# ruff: noqa: E501  (finding texts are long sentences)
"""Forensic evidence layer for images: WHERE the image looks altered and WHAT specifically is odd.

Outputs are evidence for a human reviewer, not a verdict. Each finding states what was measured, how strong it is and its known limits.

    localisation : patch-level scores from the whole-image detector (sliding windows), noise-residual and error-level (ELA) maps
    provenance   : camera/software metadata, AI-generator text chunks, content-credentials (C2PA) markers, JPEG recompression history
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np

AI_SOFTWARE = re.compile(r"stable.?diffusion|midjourney|dall.?e|firefly|comfyui|automatic1111|novelai|leonardo|ideogram|flux|runway|sora|imagen|gemini|"
                         r"chatgpt|openai|bing image|canva ai|adobe generative", re.I)
EDIT_SOFTWARE = re.compile(r"photoshop|gimp|lightroom|snapseed|facetune|meitu|picsart|pixlr|affinity|capture one|faceapp|remini", re.I)
PNG_AI_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "invokeai", "dream")


# ------------------------------------------------------------------ provenance
def provenance_findings(path: str) -> list[dict]:
    from PIL import Image
    from PIL.ExifTags import TAGS

    out: list[dict] = []
    raw = Path(path).read_bytes()
    with Image.open(path) as im:
        fmt, info = im.format, dict(im.info)
        exif = {TAGS.get(k, k): v for k, v in (im.getexif() or {}).items()}
        size = im.size
    c2 = c2pa_manifest(path)
    if c2:
        out += c2["findings"]
    elif b"c2pa" in raw or b"jumbf" in raw.lower():
        out.append({"id": "content_credentials", "strength": "moderate", "direction": "provenance",
                    "text": "Content Credentials (C2PA) markers are embedded but could not be parsed."})
    for key in PNG_AI_KEYS:
        if any(str(k).lower().startswith(key) for k in info):
            out.append({"id": "generator_text_chunk", "strength": "strong", "direction": "synthetic",
                        "text": f"PNG text chunk '{key}' is present; image generators (e.g. Stable Diffusion UIs) write prompts/settings there."})
            break
    software = str(exif.get("Software", "")) + " " + " ".join(str(v) for k, v in info.items() if str(k).lower() in ("software", "comment", "description"))
    if AI_SOFTWARE.search(software):
        out.append({"id": "ai_software_tag", "strength": "strong", "direction": "synthetic", "text": f"Software/description metadata names an AI tool: '{software.strip()[:80]}'."})
    elif EDIT_SOFTWARE.search(software):
        out.append({"id": "edit_software_tag", "strength": "moderate", "direction": "edited", "text": f"Software tag names an editing application: '{software.strip()[:80]}'."})
    if exif.get("Make") or exif.get("Model"):
        out.append({"id": "camera_metadata", "strength": "weak", "direction": "authentic",
                    "text": f"Camera metadata present ({exif.get('Make', '')} {exif.get('Model', '')}). Metadata is easy to forge, so this is weak support only."})
    elif fmt == "JPEG":
        out.append({"id": "no_camera_metadata", "strength": "weak", "direction": "neutral",
                    "text": "No camera make/model metadata. Common for images re-shared through social media or messaging, so not evidence of manipulation by itself."})
    if fmt == "JPEG":
        q = _jpeg_quality_estimate(path)
        if q is not None:
            out.append({"id": "jpeg_quality", "strength": "weak", "direction": "neutral", "text": f"JPEG quantisation suggests quality about {q}. Low quality weakens pixel-level forensics."})
    if min(size) < 256:
        out.append({"id": "low_resolution", "strength": "moderate", "direction": "neutral",
                    "text": f"Image is small ({size[0]}x{size[1]}); detector reliability drops on low-resolution images."})
    return out


AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia", "algorithmicmedia", "compositesynthetic")


def c2pa_manifest(path: str) -> Optional[dict]:
    """Parse Content Credentials with the C2PA SDK. Returns {"declared_ai": bool, "valid": str, "hash_mismatch": bool, "findings": [...]} or None."""
    try:
        import json

        import c2pa
        with c2pa.Reader(path) as r:
            doc = json.loads(r.json())
    except Exception:
        return None
    m = doc.get("manifests", {}).get(doc.get("active_manifest"), {})
    state = str(doc.get("validation_state", "Unknown"))
    status = doc.get("validation_status", []) or []
    codes = sorted({str(x.get("code")) for x in status})
    hash_mismatch = any("hash" in c.lower() and "mismatch" in c.lower() for c in codes)
    actions = []
    for a in m.get("assertions", []):
        if str(a.get("label", "")).startswith("c2pa.actions"):
            actions += a.get("data", {}).get("actions", [])
    names = sorted({str(a.get("action")) for a in actions})
    agents = sorted({str((a.get("softwareAgent") or {}).get("name", a.get("softwareAgent", ""))) for a in actions if a.get("softwareAgent")})
    sources = [str(a.get("digitalSourceType", "")).rsplit("/", 1)[-1] for a in actions if a.get("digitalSourceType")]
    declared_ai = any(x.lower() in AI_SOURCE_TYPES for x in sources)
    issuer = (m.get("signature_info") or {}).get("issuer", "unknown issuer")
    when = next((a.get("when") for a in actions if a.get("action") == "c2pa.created"), None)
    trust = ("The signature chain is trusted." if state == "Trusted" else
             f"Signature validation state: {state} ({'; '.join(sorted({str(x.get('explanation')) for x in status})) or 'no detail'}). "
             "This usually means the issuer is not on the verifier's trust list, so treat the content as a declaration, not proof.")
    tamper = (" The pixel hash does NOT match the signed manifest: the image was modified after signing." if hash_mismatch
              else " No pixel-hash mismatch was reported: the image data matches what was signed.")
    findings = []
    if declared_ai:
        findings.append({"id": "declared_ai_generated", "strength": "strong", "direction": "synthetic", "source": "c2pa",
                         "text": f"Content Credentials declare this image AI-generated ({', '.join(sources)}); created with {', '.join(agents) or 'an AI tool'}"
                                 f"{' on ' + str(when)[:10] if when else ''}, issuer {issuer}. {trust}{tamper}"})
    else:
        findings.append({"id": "content_credentials", "strength": "moderate", "direction": "provenance", "source": "c2pa",
                         "text": f"Content Credentials present (issuer {issuer}; actions: {', '.join(names) or 'none listed'}). {trust}{tamper}"})
    return {"declared_ai": declared_ai, "valid": state, "hash_mismatch": hash_mismatch, "generators": agents, "findings": findings}


def _jpeg_quality_estimate(path: str) -> Optional[int]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            tables = getattr(im, "quantization", None)
        if not tables:
            return None
        scale = float(np.array(tables[0], dtype=float).mean()) / 57.69 * 100          # IJG scale vs the standard luminance table
        return int(max(1, min(100, round((200 - scale) / 2 if scale <= 100 else 5000 / scale))))
    except Exception:
        return None


# ------------------------------------------------------------------ pixel maps
def _blocks(a: np.ndarray, b: int) -> np.ndarray:
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def noise_inconsistency_map(img, block: int = 24) -> np.ndarray:
    """Block-wise robust z-score of local noise level (log variance of the high-frequency residual). High = noise differs from the rest of the image."""
    import cv2
    g = np.asarray(img.convert("L"), dtype=np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    v = np.log(_blocks(res * res, block) + 1e-3)
    return np.abs(_robust_z(v))


def ela_map(img, quality: int = 90, block: int = 24) -> np.ndarray:
    """Error-level analysis: re-save as JPEG and measure the difference per block. Regions with a different compression history stand out."""
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    d = np.abs(np.asarray(img.convert("RGB"), dtype=np.float32) - np.asarray(rec, dtype=np.float32)).mean(axis=2)
    return np.abs(_robust_z(_blocks(d, block)))


def detector_heatmap(img, frac: float = 0.4, stride: float = 0.15) -> np.ndarray:
    """Sliding-window P(AI) from the whole-image detector, averaged over overlapping windows -> map at (about) 1/8 image resolution."""
    import torch

    from agents.image_agent.tools.modern_detector import _load
    ctx = _load()
    w, h = img.size
    ww, wh = max(64, int(w * frac)), max(64, int(h * frac))
    xs = sorted({min(x, w - ww) for x in range(0, w - ww + 1, max(1, int(w * stride)))} | {w - ww})
    ys = sorted({min(y, h - wh) for y in range(0, h - wh + 1, max(1, int(h * stride)))} | {h - wh})
    crops = [(x, y, img.crop((x, y, x + ww, y + wh))) for y in ys for x in xs]
    gw, gh = max(1, w // 8), max(1, h // 8)
    acc, cnt = np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32)
    for i in range(0, len(crops), 16):
        part = crops[i:i + 16]
        x = ctx["proc"](images=[c for _, _, c in part], return_tensors="pt")["pixel_values"].to(ctx["device"], ctx["dtype"])
        with torch.no_grad():
            lg = ctx["model"](pixel_values=x).logits.float()
        p = torch.sigmoid(lg[:, ctx["fake"]] - lg[:, ctx["real"]]).cpu().numpy()
        for (cx, cy, _), pv in zip(part, p):
            x0, x1, y0, y1 = cx // 8, (cx + ww) // 8, cy // 8, (cy + wh) // 8
            acc[y0:y1, x0:x1] += pv
            cnt[y0:y1, x0:x1] += 1
    return acc / np.maximum(cnt, 1)


# ------------------------------------------------------------------ regions + report
def top_regions(heat: np.ndarray, rel_thresh: float = 0.5, k: int = 3, min_frac: float = 0.01, abs_thresh: Optional[float] = None) -> list[dict]:
    """Connected hot regions of a map as boxes in relative image coordinates (0..1). `rel_thresh` is a share of the way from median to max."""
    import cv2
    lo, hi = float(np.median(heat)), float(heat.max())
    if hi - lo < 1e-6 or (abs_thresh is not None and hi < abs_thresh):
        return []
    mask = (heat >= (abs_thresh if abs_thresh is not None else lo + rel_thresh * (hi - lo))).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = heat.shape
    regs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area / (H * W) < min_frac:
            continue
        regs.append({"box": [round(float(x) / W, 3), round(float(y) / H, 3), round(float(x + w) / W, 3), round(float(y + h) / H, 3)],
                     "area_share": round(float(area) / (H * W), 3), "mean_score": round(float(heat[lab == i].mean()), 3)})
    return sorted(regs, key=lambda r: r["mean_score"] * r["area_share"] ** 0.5, reverse=True)[:k]


def render_overlay(img, heat: np.ndarray, out_path: str, regions: list[dict], fixed_scale: bool = False) -> str:
    import cv2
    from PIL import Image, ImageDraw
    h = cv2.resize(heat, img.size, interpolation=cv2.INTER_CUBIC)
    h = np.clip(h, 0, 1) if fixed_scale else np.clip((h - h.min()) / (h.max() - h.min() + 1e-6), 0, 1)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    over = Image.fromarray((0.55 * np.asarray(img.convert("RGB")) + 0.45 * color).astype(np.uint8))
    d = ImageDraw.Draw(over)
    W, H = img.size
    for i, r in enumerate(regions, 1):
        b = r["box"]
        d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(255, 255, 255), width=3)
        d.text((b[0] * W + 5, b[1] * H + 5), str(i), fill=(255, 255, 255))
    over.save(out_path)
    return out_path


def analyse(path: str, out_dir: Optional[str] = None, face_box: Optional[list[float]] = None) -> dict:
    """Full evidence package for one image: findings (what), regions + overlay (where). Never raises; failures are recorded as findings."""
    from PIL import Image, ImageOps

    out_dir = Path(out_dir or Path(path).parent)
    result: dict = {"findings": [], "regions": [], "evidence_map_file": None, "maps": {}, "c2pa": None}
    try:
        result["findings"] += provenance_findings(path)
        c2 = c2pa_manifest(path)
        result["c2pa"] = {k: c2[k] for k in ("declared_ai", "valid", "hash_mismatch", "generators")} if c2 else None
    except Exception as exc:
        result["findings"].append({"id": "provenance_failed", "strength": "weak", "direction": "neutral", "text": f"Metadata check failed: {type(exc).__name__}."})
    try:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        from agents.image_agent.tools.edit_localizer import prob_map
        heat = prob_map(img)                                   # P(pixel was AI-edited/synthesised), validated pixel-level localiser
        area = float((heat >= 0.5).mean())
        result["maps"]["edit"] = {"max": round(float(heat.max()), 3), "edited_area": round(area, 4)}
        regs = top_regions(heat, abs_thresh=0.5)
        result["regions"] = regs
        result["evidence_map_file"] = Path(render_overlay(img, heat, str(out_dir / "evidence_map.png"), regs, fixed_scale=True)).name
        if regs and area >= 0.005:
            r = regs[0]
            big = area > 0.6
            result["findings"].append({
                "id": "ai_edit_pixels" if not big else "mostly_synthetic_pixels", "strength": "strong" if heat.max() >= 0.9 else "moderate",
                "direction": "synthetic", "region": r["box"],
                "text": (f"The edit localiser marks {area * 100:.1f}% of the image as AI-generated/edited pixels (peak probability {heat.max():.2f}); "
                         f"the largest region is shown as region 1 on the evidence map. " if not big else
                         f"The edit localiser marks {area * 100:.0f}% of the image as synthetic pixels, i.e. most of the picture rather than one edit. ")
                        + "Validated on Stable-Diffusion-style inpainting and text-to-image (about 97% / 91% flagged, 0.7% of clean photos flagged). "
                          "Limits: not validated on other editors (Photoshop generative fill, Flux, GPT-image); does not detect face swaps; "
                          "clean photos re-saved at JPEG q75 are flagged about 9% of the time."})
        nz = noise_inconsistency_map(img)
        zmax = float(np.percentile(nz, 99))
        result["maps"]["noise_p99_z"] = round(zmax, 2)
        if zmax >= 4.0:
            result["findings"].append({"id": "noise_inconsistency", "strength": "weak", "direction": "edited",
                                       "text": f"Noise level varies strongly between regions (99th-percentile robust z = {zmax:.1f}); can indicate splicing or local retouching. "
                                               "Also caused by natural texture changes, so supporting evidence only."})
    except Exception as exc:
        result["findings"].append({"id": "localisation_failed", "strength": "weak", "direction": "neutral", "text": f"Localisation could not run: {type(exc).__name__}: {exc}"})
    return result
