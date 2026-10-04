"""Score fusion and verdict logic. Pure functions, no I/O, shared by the live pipeline and the offline benchmark.

    usable   = tools that did not error and have confidence >= min_tool_confidence
    weight_i = base_weight[tool] * length_factor(tool, n_words) * confidence_i
    fused    = sum(weight_i * score_i) / sum(weight_i)
    spread   = max(score) - min(score) over usable tools
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.calibration import load_calibration
from agents.schemas import ToolVerdict


def length_factor(tool_name: str, n_words: int) -> float:
    """Down-weight tools that are unreliable on short text (text modality only)."""
    if tool_name == "text_slm":
        return 0.5 if n_words < 100 else 1.0
    if tool_name == "text_dl":
        return 0.7 if n_words < 50 else 1.0
    return 1.0


@dataclass
class FusionResult:
    fused: Optional[float]
    spread: float
    usable: list[str]
    weights: dict[str, float]
    verdict: str
    confidence: float
    escalate: bool
    needs_reflexion: bool
    notes: list[str] = field(default_factory=list)


def fuse(verdicts: list[ToolVerdict], modality: str, n_words: int = 0, cfg: Optional[dict] = None,
         retries: int = 0) -> FusionResult:
    cfg = cfg or load_calibration()
    min_conf = float(cfg["min_tool_confidence"])
    spread_limit = float(cfg["spread_threshold"])
    th = cfg["thresholds"][modality]

    usable = [v for v in verdicts if not v.error and v.confidence >= min_conf]
    notes: list[str] = []
    skipped = [v.tool_name for v in verdicts if v not in usable]
    if skipped:
        notes.append(f"Excluded from fusion (error or low confidence): {', '.join(skipped)}.")

    if not usable:
        notes.append("No usable tool verdicts.")
        return FusionResult(None, 0.0, [], {}, "uncertain", 0.0, True, False, notes)

    weights = {
        v.tool_name: cfg["weights"].get(v.tool_name, 1.0) * length_factor(v.tool_name, n_words) * max(v.confidence, 1e-6)
        for v in usable
    }
    total = sum(weights.values())
    fused = sum(weights[v.tool_name] * v.score for v in usable) / total
    scores = [v.score for v in usable]
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0

    if fused >= th["t_hi"]:
        verdict = "synthetic"
    elif fused <= th["t_lo"]:
        verdict = "authentic"
    else:
        verdict = "uncertain"

    confidence = abs(fused - 0.5) * 2.0
    disagreement = spread > spread_limit
    if disagreement:
        confidence -= 0.15
        notes.append(
            f"High inter-tool disagreement (spread={spread:.2f}): "
            + ", ".join(f"{v.tool_name}={v.score:.2f}" for v in usable) + "."
        )
    confidence = max(0.0, min(0.95, confidence))

    if len(usable) < 2:
        notes.append("Fewer than two usable tools; verdict rests on a single detector.")
    if verdict == "uncertain":
        notes.append("Fused score lies inside the uncertain band.")

    escalate = len(usable) < 2 or verdict == "uncertain" or disagreement
    needs_reflexion = len(usable) >= 2 and disagreement and retries == 0
    return FusionResult(fused, spread, [v.tool_name for v in usable], weights, verdict, confidence,
                        escalate, needs_reflexion, notes)
