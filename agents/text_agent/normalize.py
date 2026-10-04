"""Input normalisation for the text tools.

Homoglyph / invisible-character attacks (e.g. Latin 'a' -> Cyrillic 'а', zero-width spaces) leave the text looking
identical to a human but shatter the tokenizer, so a classifier trained on clean text loses most of its signal
(DeBERTa v2: AUC 0.72 on RAID homoglyph vs 0.95-0.97 on synonym / paraphrase attacks).
Canonicalising the input first removes the attack. The *amount* of canonicalisation is itself evidence, so it is
reported as a feature instead of being thrown away.
"""
from __future__ import annotations

import re
import unicodedata

# Characters that render like ASCII letters but come from other scripts (Cyrillic, Greek, a few Armenian/Cherokee).
_CONFUSABLES = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ɡ": "g", "ӏ": "l", "ь": "b", "п": "n", "т": "t", "к": "k", "м": "m",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Ү": "Y",
    # Greek
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "ι": "i", "κ": "k", "υ": "u", "χ": "x", "γ": "y",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin look-alikes
    "ı": "i", "ɑ": "a", "ƅ": "b", "ɩ": "i", "ⅼ": "l", "ǀ": "l", "ʋ": "v", "ᴠ": "v", "ᴡ": "w",
    "ᴢ": "z", "ꓲ": "I", "ꓳ": "O",
}
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)
_SPACES = re.compile(r"[ \t  -   　]+")
_TRANS = {ord(k): v for k, v in _CONFUSABLES.items()}


def normalize_text(text: str) -> tuple[str, dict]:
    """Return (canonical text, stats). Stats feed the report as evidence of tampering."""
    n_chars = max(len(text), 1)
    zero_width = sum(1 for ch in text if ord(ch) in _ZERO_WIDTH)
    out = text.translate(_ZERO_WIDTH)
    out = unicodedata.normalize("NFKC", out)               # full-width forms, ligatures, compatibility variants
    # Only remap when the document is mostly Latin; genuine Russian / Greek text must not be mangled.
    letters = [ch for ch in out if ch.isalpha()]
    latin_share = sum(1 for ch in letters if ch.isascii()) / max(len(letters), 1)
    confusable = 0
    if latin_share >= 0.5:
        confusable = sum(1 for ch in out if ord(ch) in _TRANS)
        out = out.translate(_TRANS)
    out = _SPACES.sub(" ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    changed = sum(1 for a, b in zip(text, out) if a != b) + abs(len(text) - len(out))
    return out, {
        "homoglyph_chars": confusable,
        "zero_width_chars": zero_width,
        "tamper_ratio": round(min((confusable + zero_width) / n_chars, 1.0), 4),
        "changed_chars": changed,
    }
