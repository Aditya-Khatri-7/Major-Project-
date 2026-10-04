"""LangGraph orchestrator.

    START -> route(modality) -> text_agent | image_agent -> rag -> verifier
    verifier -> (tools disagree and no retry used yet) -> reflexion -> verifier
    verifier -> report -> persist -> END
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agents.common import log
from agents.image_agent.agent import run_image_agent
from agents.reflexion import reflexion_node
from agents.report import build_report
from agents.schemas import ForensicState
from agents.text_agent.agent import run_text_agent
from agents.verifier_agent import run_verifier
from observability import log_event
from rag.retriever import run_rag_node


def route_by_modality(state: ForensicState) -> str:
    return "text_agent" if state["modality"] == "text" else "image_agent"


def route_after_verifier(state: ForensicState) -> str:
    return "reflexion" if state.get("needs_reflexion") and state.get("retries", 0) == 0 else "report"


def persist_evidence(state: ForensicState) -> dict:
    """Write the audit record and the case-memory entry. Failures are logged, never raised."""
    try:
        from evidence_store.models import save_record
        save_record(state)
    except Exception as exc:
        log_event(log, "evidence_store_failed", level=logging.ERROR, job_id=state["job_id"], error=str(exc))
    try:
        from rag.case_memory import add_case
        add_case(state)
    except Exception as exc:
        log_event(log, "case_memory_failed", level=logging.ERROR, job_id=state["job_id"], error=str(exc))
    return {"report": state.get("report")}


def build_graph():
    g = StateGraph(ForensicState)
    g.add_node("text_agent", run_text_agent)
    g.add_node("image_agent", run_image_agent)
    g.add_node("rag", run_rag_node)
    g.add_node("verifier", run_verifier)
    g.add_node("reflexion", reflexion_node)
    g.add_node("report", build_report)
    g.add_node("persist", persist_evidence)

    g.add_conditional_edges(START, route_by_modality, {"text_agent": "text_agent", "image_agent": "image_agent"})
    g.add_edge("text_agent", "rag")
    g.add_edge("image_agent", "rag")
    g.add_edge("rag", "verifier")
    g.add_conditional_edges("verifier", route_after_verifier, {"reflexion": "reflexion", "report": "report"})
    g.add_edge("reflexion", "verifier")
    g.add_edge("report", "persist")
    g.add_edge("persist", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


if __name__ == "__main__":
    import json
    import uuid

    from agents.schemas import initial_state

    demo = (
        "This essay explores the fundamental principles of artificial intelligence and its "
        "applications in modern society. Furthermore, it is important to note that AI systems "
        "have become increasingly prevalent across many industries and sectors of the economy."
    )
    out = get_graph().invoke(initial_state(str(uuid.uuid4()), "text", demo))
    print(json.dumps(out["report"], indent=2))
