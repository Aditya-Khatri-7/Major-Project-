"""Input validation for the API: text length, image type (by content, not by name), size and dimensions."""
from __future__ import annotations

import io

from config import settings

MAX_PIXELS = 50_000_000
_EXTENSIONS = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}


class InputRejected(ValueError):
    """Raised for invalid user input; `status` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_text(text: str) -> str:
    cleaned = (text or "").replace("\x00", "").strip()
    n_words = len(cleaned.split())
    if n_words < settings.min_text_words:
        raise InputRejected(f"Text must contain at least {settings.min_text_words} words (got {n_words}).")
    if n_words > settings.max_text_words:
        raise InputRejected(f"Text must contain at most {settings.max_text_words} words (got {n_words}).", 413)
    return cleaned


def sniff_image_type(head: bytes) -> str | None:
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_bytes(data: bytes) -> str:
    """Check size, real file type and dimensions. Returns the file extension to store it under."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise InputRejected("Empty file.")
    if len(data) > settings.max_image_mb * 1024 * 1024:
        raise InputRejected(f"Image larger than {settings.max_image_mb} MB.", 413)
    kind = sniff_image_type(data[:16])
    if kind is None:
        raise InputRejected("Unsupported file type. Upload a JPEG, PNG or WebP image.", 415)
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            if (img.format or "").lower() != kind:
                raise InputRejected("File content does not match its declared image type.", 415)
            if max(width, height) > settings.max_image_side_px or width * height > MAX_PIXELS:
                raise InputRejected(f"Image dimensions exceed {settings.max_image_side_px}px on a side.", 413)
            img.verify()
    except InputRejected:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InputRejected(f"Corrupt or unreadable image: {exc}", 415) from exc
    return _EXTENSIONS[kind]
