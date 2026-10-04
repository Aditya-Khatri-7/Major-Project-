"""Verifier / critic node: fuses tool verdicts, checks disagreement, decides on Reflexion and escalation.

RAG never changes the score. Its only effect on the decision: a *human-verified* case neighbour
(similarity >= 0.85) that contradicts the verdict forces escalation to a human.
"""
from __future__ import annotations

from agents.common import JUDGE_TOOLS, log
from agents.fusion import fuse
from agents.schemas import ForensicState
from config import settings
from observability import log_event

NEIGHBOUR_SIMILARITY = 0.85


def _judge_can_retry(state: ForensicState) -> bool:
    judge = JUDGE_TOOLS[state["modality"]]
    has_working_judge = any(v.tool_name == judge and not v.error for v in state["tool_verdicts"])
    return has_working_judge and bool(settings.anthropic_api_key)


def run_verifier(state: ForensicState) -> dict:
    modality = state["modality"]
    n_words = len(state["input_ref"].split()) if modality == "text" else 0
    result = fuse(state["tool_verdicts"], modality, n_words, retries=state.get("retries", 0))

    notes = list(result.notes)
    verdict, escalate = result.verdict, result.escalate

    for ev in state.get("retrieved_evidence", []):
        if ev.evidence_type != "case_study" or not ev.verified or ev.similarity_score < NEIGHBOUR_SIMILARITY:
            continue
        if ev.label in ("authentic", "synthetic") and verdict in ("authentic", "synthetic"):
            if ev.label != verdict:
                escalate = True
                notes.append(f"Contradicts verified case {ev.source_id} (human label: {ev.label}).")
            else:
                notes.append(f"Consistent with verified case {ev.source_id}.")

    needs_reflexion = result.needs_reflexion and settings.enable_reflexion
    if needs_reflexion and not _judge_can_retry(state):
        needs_reflexion = False
        notes.append("Reflexion skipped: no working judge tool available.")

    log_event(log, "verifier_done", job_id=state["job_id"], fused=result.fused, verdict=verdict,
              spread=round(result.spread, 3), escalate=escalate, needs_reflexion=needs_reflexion)
    return {
        "fused_score": result.fused,
        "final_verdict": verdict,
        "final_confidence": round(result.confidence, 4),
        "escalate_to_human": escalate,
        "verifier_notes": " | ".join(notes) if notes else "Tools in agreement.",
        "needs_reflexion": needs_reflexion,
    }
