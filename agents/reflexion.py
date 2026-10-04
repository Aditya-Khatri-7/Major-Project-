"""Reflexion step: when tools disagree, re-run the judge (LLM / VLM) once with peer outputs and evidence."""
from __future__ import annotations

from agents.common import JUDGE_TOOLS, log
from agents.schemas import ForensicState, RetrievedEvidence, ToolVerdict
from observability import log_event


def peer_summary(verdicts: list[ToolVerdict], judge_name: str) -> str:
    lines = [
        f"- {v.tool_name}: P(synthetic)={v.score:.2f}, confidence={v.confidence:.2f}. {v.explanation}"
        for v in verdicts if v.tool_name != judge_name and not v.error
    ]
    return "\n".join(lines) or "(no other detector output)"


def evidence_summary(evidence: list[RetrievedEvidence], limit: int = 4) -> str:
    return "\n".join(f"- [{e.source_id}] {e.content_snippet[:300]}" for e in evidence[:limit])


def rerun_judge(modality: str, input_ref: str, verdicts: list[ToolVerdict],
                evidence: list[RetrievedEvidence], client=None) -> list[ToolVerdict]:
    """Return the verdict list with the judge verdict replaced by its Reflexion re-check (if that succeeds)."""
    judge_name = JUDGE_TOOLS[modality]
    peers = peer_summary(verdicts, judge_name)
    notes = evidence_summary(evidence)
    if modality == "text":
        from agents.text_agent.tools import llm_judge
        new = llm_judge.score_text_with_context(input_ref, peers, notes, client=client)
    else:
        from agents.image_agent.tools import vlm_judge
        new = vlm_judge.score_image_with_context(input_ref, peers, notes, client=client)

    out = []
    for v in verdicts:
        if v.tool_name == judge_name and not new.error:
            previous = v.score
            out.append(new.model_copy(update={"raw_features": {**new.raw_features, "previous_score": previous}}))
        else:
            out.append(v)
    return out


def reflexion_node(state: ForensicState) -> dict:
    updated = rerun_judge(state["modality"], state["input_ref"], state["tool_verdicts"],
                          state.get("retrieved_evidence", []))
    log_event(log, "reflexion_done", job_id=state["job_id"])
    return {"tool_verdicts": updated, "retries": state.get("retries", 0) + 1}
