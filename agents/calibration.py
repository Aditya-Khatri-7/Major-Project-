"""Calibration parameters (written by training/calibrate.py) and helpers to apply them.

All thresholds and fusion weights live in one JSON file so they are set in exactly one place,
on held-out calibration data, and are frozen before any test set is evaluated.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Optional

from agents.schemas import ToolVerdict
from config import settings

DEFAULTS: dict = {
    "text_dl": {"temperature": 1.0},
    "text_slm": {"a": -10.0, "b": 9.0, "calibrated": False},   # p = sigmoid(a * binoculars + b)
    "image_dl": {"temperature": 1.0},
    "image_clip": {"temperature": 1.0},
    "image_general": {"temperature": 1.0},
    "weights": {"text_dl": 1.0, "text_slm": 1.0, "text_llm": 1.0, "image_dl": 1.0, "image_clip": 1.0, "image_general": 1.0, "image_vlm": 1.0},
    "thresholds": {
        "text": {"t_lo": 0.35, "t_hi": 0.65, "t_star": 0.5},
        "image": {"t_lo": 0.35, "t_hi": 0.65, "t_star": 0.5},
    },
    "spread_threshold": 0.35,
    "min_tool_confidence": 0.15,
}

_cache: dict = {}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_calibration(path: Optional[str] = None) -> dict:
    """Defaults merged with the calibration file (if it exists). Cached by file mtime."""
    p = Path(path or settings.calibration_path)
    mtime = p.stat().st_mtime if p.exists() else None
    key = (str(p), mtime)
    if key in _cache:
        return _cache[key]
    cfg = copy.deepcopy(DEFAULTS)
    if p.exists():
        cfg = _merge(cfg, json.loads(p.read_text(encoding="utf-8")))
    _cache.clear()
    _cache[key] = cfg
    return cfg


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def tool_probability(tool_name: str, raw: dict, cal: dict) -> Optional[float]:
    """Calibrated P(AI) from a tool's raw output, or None if the tool has no calibration model."""
    if tool_name in ("text_dl", "image_dl", "image_clip", "image_general") and "raw_logit" in raw:
        temperature = max(float(cal[tool_name].get("temperature", 1.0)), 1e-3)
        return sigmoid(float(raw["raw_logit"]) / temperature)
    if tool_name == "text_slm" and "binoculars_score" in raw:
        c = cal["text_slm"]
        return sigmoid(float(c["a"]) * float(raw["binoculars_score"]) + float(c["b"]))
    return None


def tool_confidence(p: float, raw: dict) -> float:
    return float(min(0.95, abs(p - 0.5) * 2.0 * float(raw.get("conf_factor", 1.0))))


def recalibrate(verdict: ToolVerdict, cal: dict) -> ToolVerdict:
    """Re-derive score/confidence from raw features under the given calibration.

    Lets cached benchmark scores follow the latest calibration without re-running any model.
    Judge tools (LLM / VLM) and errored verdicts are returned unchanged.
    """
    if verdict.error:
        return verdict
    p = tool_probability(verdict.tool_name, verdict.raw_features, cal)
    if p is None:
        return verdict
    conf_factor_raw = dict(verdict.raw_features)
    if verdict.tool_name == "text_slm":
        conf_factor_raw["conf_factor"] = 1.0 if cal["text_slm"].get("calibrated", True) else 0.5
    return verdict.model_copy(update={"score": p, "confidence": tool_confidence(p, conf_factor_raw)})
