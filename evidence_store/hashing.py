"""SHA-256 hashing of analysed inputs, for the audit trail."""
from __future__ import annotations

import hashlib
from pathlib import Path


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_input(input_ref: str, modality: str) -> str:
    """Hash the file bytes for images, the UTF-8 text for text. The modality decides; never guess from the string."""
    if modality == "image":
        return hash_bytes(Path(input_ref).read_bytes())
    return hash_bytes(input_ref.encode("utf-8"))
