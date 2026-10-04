"""RAG layer 3: case memory. Every analysed job is stored; human corrections mark it verified.

Only tool explanations and the verdict are embedded, never the raw analysed content.
"""
from __future__ import annotations

from agents.schemas import ForensicState
from rag.store import get_cases


def case_document(state: ForensicState) -> str:
    lines = [f"{state['modality']} case. Verdict: {state.get('final_verdict')}."]
    for v in state.get("tool_verdicts", []):
        if not v.error:
            lines.append(f"{v.tool_name}: {v.explanation[:160]}")
    return " ".join(lines)


def add_case(state: ForensicState) -> None:
    if not state.get("final_verdict"):
        return
    get_cases().upsert(
        ids=[state["job_id"]],
        documents=[case_document(state)],
        metadatas=[{
            "job_id": state["job_id"], "modality": state["modality"], "label": state["final_verdict"],
            "verified": False, "fused_score": float(state.get("fused_score") or 0.0),
        }],
    )


def apply_correction(job_id: str, label: str) -> bool:
    """Mark a stored case as human-verified with the corrected label. False if the case is unknown."""
    col = get_cases()
    existing = col.get(ids=[job_id])
    if not existing["ids"]:
        return False
    meta = dict(existing["metadatas"][0])
    meta.update({"label": label, "verified": True})
    col.update(ids=[job_id], metadatas=[meta])
    return True
