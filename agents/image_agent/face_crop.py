"""Crop the main face out of a full photo before the face-deepfake tools see it.

The face tools were trained on tight face crops (30% margin). On a normal photo with background they gave 24-52% false alarms
on real faces (eval/stress_image.py, pad_1.8x_context). Images that are already tight crops pass through unchanged.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from agents.common import log
from observability import log_event

MARGIN = 0.3
MAX_FACE_AREA_FRACTION = 0.25     # face box larger than this share of the image = already a crop (a 30%-margin crop is about 0.39)
MIN_SIDE_PX = 40
MIN_FACE_PROB = 0.9
DETECT_MAX_SIDE = 1024

_lock = threading.Lock()
_mtcnn = None


def _detector():
    global _mtcnn
    with _lock:
        if _mtcnn is None:
            import torch
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu", post_process=False)
        return _mtcnn


def crop_face_if_needed(image_path: str) -> Optional[str]:
    """Return the path of a face crop written next to the input, or None when the image should be used as is."""
    try:
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(image_path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        det = _detector()
        rotated = 0
        full = img
        scale = min(1.0, DETECT_MAX_SIDE / max(img.size))
        if scale < 1.0:                    # detect on a downscaled copy (4 passes on a 6000 px photo took seconds); crop from the full image
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        boxes, probs = det.detect(img)
        if boxes is None or len(boxes) == 0 or float(max(probs)) < MIN_FACE_PROB:
            # Sideways / upside-down photo (stress test: 90-degree rotation gave 64-74% false alarms): find the upright orientation.
            for angle in (90, 270, 180):
                cand = img.rotate(angle, expand=True)
                b, pr = det.detect(cand)
                if b is not None and len(b) and float(max(pr)) >= MIN_FACE_PROB:
                    img, boxes, probs, rotated = cand, b, pr, angle
                    break
        if boxes is None or len(boxes) == 0:
            return None
        if scale < 1.0:                                       # map boxes back to full-resolution coordinates, in the (possibly rotated) orientation
            full = full.rotate(rotated, expand=True) if rotated else full
            boxes = boxes / scale
            img = full
        w, h = img.size
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if not rotated and (x2 - x1) * (y2 - y1) / float(w * h) >= MAX_FACE_AREA_FRACTION:
            return None
        side = max(x2 - x1, y2 - y1) * (1 + 2 * MARGIN)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, top = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        right, bottom = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        if right - left < MIN_SIDE_PX or bottom - top < MIN_SIDE_PX:
            return None
        src = Path(image_path)
        out = src.with_name(f"{src.stem}_facecrop.png")
        Image.fromarray(np.asarray(img)[top:bottom, left:right]).save(out)
        log_event(log, "face_cropped", box=[int(left), int(top), int(right), int(bottom)], source_size=[w, h], rotated=rotated)
        return str(out)
    except Exception as exc:                                  # cropping must never break the pipeline
        log_event(log, "face_crop_failed", error=f"{type(exc).__name__}: {exc}")
        return None


def face_fraction(image_path: str) -> float:
    """Share of the image area covered by the largest detected face (0.0 when no face is found). Used to tell tight face crops from scenes."""
    try:
        from PIL import Image, ImageOps
        with Image.open(image_path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        scale = min(1.0, DETECT_MAX_SIDE / max(img.size))
        if scale < 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        boxes, probs = _detector().detect(img)
        if boxes is None or len(boxes) == 0 or float(max(probs)) < MIN_FACE_PROB:
            return 0.0
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return float(min(1.0, (x2 - x1) * (y2 - y1) / float(img.width * img.height)))
    except Exception:
        return 0.0
