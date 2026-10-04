"""Report generator: structured JSON plus a deterministic human-readable summary. Every claim is citable."""
from __future__ import annotations

from pathlib import Path

from agents.common import model_versions
from agents.schemas import ForensicState, ToolVerdict

_PUBLIC_FEATURES = ("raw_logit", "n_windows", "binoculars_score", "n_tokens", "artifacts_found",
                    "reflexion", "previous_score", "tta_views")


def _public_features(v: ToolVerdict) -> dict:
    out = {k: v.raw_features[k] for k in _PUBLIC_FEATURES if k in v.raw_features}
    if "gradcam_path" in v.raw_features:
        out["gradcam_file"] = Path(v.raw_features["gradcam_path"]).name
    return out


def _summary(state: ForensicState, citations: list[str]) -> str:
    verdict = (state["final_verdict"] or "uncertain").upper()
    confidence = (state["final_confidence"] or 0.0) * 100
    fused = state.get("fused_score")
    parts = [f"Assessed as {verdict} (confidence {confidence:.0f}%"
             + (f", fused P(synthetic)={fused:.2f}" if fused is not None else "") + ")."]
    strongest = max((v for v in state["tool_verdicts"] if not v.error), key=lambda v: v.confidence, default=None)
    if strongest:
        parts.append(f"Strongest signal: {strongest.tool_name} - {strongest.explanation}")
    if state["escalate_to_human"]:
        parts.append("Human review recommended.")
    parts.append(f"Supporting sources: {', '.join(citations)}." if citations
                 else "No supporting evidence retrieved from the knowledge base.")
    return " ".join(parts)


def build_report(state: ForensicState) -> dict:
    evidence = state.get("retrieved_evidence", [])
    citations = sorted({e.source_id for e in evidence})
    report = {
        "job_id": state["job_id"],
        "modality": state["modality"],
        "final_verdict": state["final_verdict"],
        "final_confidence": state["final_confidence"],
        "fused_score": state.get("fused_score"),
        "escalate": state["escalate_to_human"],
        "verifier_notes": state.get("verifier_notes") or "",
        "reflexion_used": state.get("retries", 0) > 0,
        "tool_verdicts": [
            {"tool": v.tool_name, "score": round(v.score, 4), "confidence": round(v.confidence, 4),
             "explanation": v.explanation, "error": v.error, "features": _public_features(v)}
            for v in state["tool_verdicts"]
        ],
        "evidence": [e.model_dump() for e in evidence],
        "citations": citations,
        "summary": _summary(state, citations),
        "model_versions": model_versions(),
    }
    return {"report": report, "citations": citations}
