"""Small helpers shared by the agents."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Callable

from agents.schemas import Modality, ToolVerdict
from config import settings
from observability import get_logger, log_event

log = get_logger("agents")

TEXT_TOOL_NAMES = {"dl": "text_dl", "slm": "text_slm", "llm": "text_llm"}
IMAGE_TOOL_NAMES = {"dl": "image_dl", "clip": "image_clip", "general": "image_general", "vlm": "image_vlm"}
JUDGE_TOOLS = {"text": "text_llm", "image": "image_vlm"}


def run_tool(tool_name: str, modality: Modality, fn: Callable[..., ToolVerdict], *args, **kwargs) -> ToolVerdict:
    """Run one tool. A crash becomes an error verdict instead of taking the pipeline down."""
    start = time.perf_counter()
    try:
        verdict = fn(*args, **kwargs)
    except Exception as exc:
        log_event(log, "tool_crashed", level=logging.ERROR, tool=tool_name, error=f"{type(exc).__name__}: {exc}")
        verdict = ToolVerdict.failure(tool_name, modality, f"{type(exc).__name__}: {exc}")
    log_event(
        log, "tool_done", tool=tool_name, score=round(verdict.score, 4), confidence=round(verdict.confidence, 4),
        error=verdict.error, latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return verdict


def model_versions() -> dict:
    """Identifiers of every model / config that influenced a verdict (stored with each record)."""
    calibration = Path(settings.calibration_path)
    cal_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()[:12] if calibration.exists() else None
    return {
        "llm": f"{settings.llm_provider}:{settings.judge_model_id}",
        "text_dl": settings.text_dl_model_path,
        "slm_observer": settings.slm_observer_model,
        "slm_performer": settings.slm_performer_model,
        "image_dl": settings.image_model_path,
        "calibration_sha256": cal_hash,
    }
