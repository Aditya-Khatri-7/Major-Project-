"""Score fusion and verdict logic. Pure functions, no I/O, shared by the live pipeline and the offline benchmark.

    usable   = tools that did not error and have confidence >= min_tool_confidence
    weight_i = base_weight[tool] * length_factor(tool, n_words) * confidence_i
    fused    = sum(weight_i * score_i) / sum(weight_i)
    spread   = max(score) - min(score) over usable tools
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.calibration import load_calibration, sigmoid
from agents.schemas import ToolVerdict


SHORT_TEXT_WORDS = 60


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
    stack = (cfg.get("stack") or {}).get(modality)
    if stack:
        # Learned combination of the tools' raw logits (training/calibrate_image_stack.py). Replaces the plain average because the tools are
        # reliable on different image types (face crops vs whole scenes), so a fixed average mixes a good and a useless opinion.
        z = float(stack["bias"])
        for name, coef in stack["coef"].items():
            v = next((u for u in verdicts if u.tool_name == name and not u.error and "raw_logit" in u.raw_features), None)
            if v is not None:
                z += float(coef) * float(v.raw_features["raw_logit"]) + float(stack["present"][name])
        fused = sigmoid(z)
        notes.append("Image score from the learned tool combination (see models/calibration.json 'stack').")
    # Disagreement is judged among tools that carry real weight; advisory (low-weight) tools are only mentioned.
    min_w = float(cfg.get("spread_min_weight", 0.0))
    voters = [v for v in usable if cfg["weights"].get(v.tool_name, 1.0) >= min_w]
    advisory = [v for v in usable if v not in voters]
    scores = [v.score for v in voters]
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    if stack:
        spread = 0.0                       # the stack already weighs the tools; raw disagreement alone is not escalated
    for v in advisory:
        if abs(v.score - fused) > spread_limit:
            notes.append(f"Advisory tool {v.tool_name} (low weight) disagrees: {v.score:.2f} vs fused {fused:.2f}.")

    if fused >= th["t_hi"]:
        verdict = "synthetic"
    elif fused <= th["t_lo"]:
        verdict = "authentic"
    else:
        verdict = "uncertain"

    short_text = modality == "text" and 0 < n_words < SHORT_TEXT_WORDS
    if short_text and verdict != "uncertain":
        # Stress test: under ~60 words the classifier is near chance (AUC 0.55 at 25 words) and flags most human texts.
        verdict = "uncertain"
        notes.append(f"Text has only {n_words} words (< {SHORT_TEXT_WORDS}); too short for a reliable verdict, score is indicative only.")

    confidence = abs(fused - 0.5) * 2.0
    if short_text:
        confidence = min(confidence, 0.3)
    disagreement = spread > spread_limit
    if disagreement:
        confidence -= 0.15
        notes.append(
            f"High inter-tool disagreement (spread={spread:.2f}): "
            + ", ".join(f"{v.tool_name}={v.score:.2f}" for v in voters) + "."
        )
    confidence = max(0.0, min(0.95, confidence))

    if len(usable) < 2:
        notes.append("Fewer than two usable tools; verdict rests on a single detector.")
    elif len(voters) < 2:
        notes.append("Only one weighted tool; the others are advisory.")
    if verdict == "uncertain":
        notes.append("Fused score lies inside the uncertain band.")

    escalate = len(usable) < 2 or verdict == "uncertain" or disagreement
    needs_reflexion = len(usable) >= 2 and disagreement and retries == 0
    prov = next((v for v in usable if v.tool_name == "image_provenance" and v.score >= 0.95), None)
    if prov is not None:
        # Provenance beats statistics: the file itself says it was AI-generated. Detectors that disagree are noted, not averaged in.
        others = [f"{v.tool_name}={v.score:.2f}" for v in usable if v is not prov]
        notes.append("Provenance rule: embedded Content Credentials declare AI generation; this overrides the detector average"
                     + (f" (detectors: {', '.join(others)})." if others else "."))
        fused, verdict, confidence = max(fused, 0.97), "synthetic", 0.9
        escalate, needs_reflexion = bool(prov.raw_features.get("hash_mismatch")), False
    return FusionResult(fused, spread, [v.tool_name for v in usable], weights, verdict, confidence,
                        escalate, needs_reflexion, notes)
