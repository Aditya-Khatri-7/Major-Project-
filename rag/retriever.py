"""RAG retriever: knowledge-base and case-memory search with CRAG-lite self-correction.

Hits below MIN_SIMILARITY are dropped. If nothing survives, the query is rewritten from the tool
names (a different phrasing) and retried once. If still empty, the report says so explicitly.
"""
from __future__ import annotations

import logging
from typing import Optional

from agents.common import log
from agents.schemas import ForensicState, RetrievedEvidence, ToolVerdict
from observability import log_event
from rag.store import get_cases, get_kb

KB_RESULTS = 4
CASE_RESULTS = 2
MIN_SIMILARITY = 0.5

_TOOL_CONCEPTS = {
    "text_dl": "fine-tuned transformer classifier for AI-generated text detection",
    "text_slm": "zero-shot perplexity and cross-perplexity Binoculars detection",
    "text_llm": "LLM judgement of generic phrasing and AI writing style",
    "image_dl": "CNN artifact detector for deepfake and generated images",
    "image_vlm": "vision-language reasoning about lighting, anatomy and physical plausibility",
}


def build_query(modality: str, verdicts: list[ToolVerdict]) -> str:
    usable = sorted((v for v in verdicts if not v.error), key=lambda v: v.confidence, reverse=True)[:3]
    parts = [modality] + [v.explanation[:200] for v in usable]
    for v in usable:
        parts.extend(str(a) for a in v.raw_features.get("artifacts_found", [])[:5])
    return " ".join(parts)


def rewrite_query(modality: str, verdicts: list[ToolVerdict]) -> str:
    concepts = [_TOOL_CONCEPTS.get(v.tool_name, v.tool_name) for v in verdicts if not v.error]
    return f"known characteristics and detection of AI-generated {modality}: " + "; ".join(concepts)


def _similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))     # cosine distance -> similarity


def _kb_hits(query: str, modality: str, n: int, min_similarity: float) -> list[RetrievedEvidence]:
    if n <= 0:
        return []
    col = get_kb()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(n, col.count()),
                    where={"modality": {"$in": [modality, "general"]}})
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = _similarity(dist)
        if sim >= min_similarity:
            out.append(RetrievedEvidence(
                source_id=meta["source"], content_snippet=doc[:400], similarity_score=round(sim, 4),
                evidence_type=meta.get("evidence_type", "literature"),
            ))
    return out


def _case_hits(query: str, modality: str, n: int, min_similarity: float) -> list[RetrievedEvidence]:
    if n <= 0:
        return []
    col = get_cases()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(n, col.count()), where={"modality": modality})
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = _similarity(dist)
        if sim >= min_similarity:
            out.append(RetrievedEvidence(
                source_id=f"case:{meta.get('job_id', 'unknown')}", content_snippet=doc[:400],
                similarity_score=round(sim, 4), evidence_type="case_study",
                verified=bool(meta.get("verified", False)), label=meta.get("label"),
            ))
    return out


def retrieve_evidence(query: str, modality: str, kb_n: int = KB_RESULTS, case_n: int = CASE_RESULTS,
                      min_similarity: float = MIN_SIMILARITY) -> list[RetrievedEvidence]:
    hits = _kb_hits(query, modality, kb_n, min_similarity) + _case_hits(query, modality, case_n, min_similarity)
    return sorted(hits, key=lambda e: e.similarity_score, reverse=True)


def run_rag_node(state: ForensicState) -> dict:
    """LangGraph node. Retrieval problems never fail a job; they yield an empty evidence list."""
    verdicts = state.get("tool_verdicts", [])
    if not any(not v.error for v in verdicts):
        return {"retrieved_evidence": []}
    modality = state["modality"]
    try:
        evidence: Optional[list[RetrievedEvidence]] = retrieve_evidence(build_query(modality, verdicts), modality)
        if not evidence:                                      # CRAG-lite: one rewritten retry
            evidence = retrieve_evidence(rewrite_query(modality, verdicts), modality)
    except Exception as exc:
        log_event(log, "rag_failed", level=logging.WARNING, job_id=state["job_id"], error=str(exc))
        evidence = []
    log_event(log, "rag_done", job_id=state["job_id"], hits=len(evidence))
    return {"retrieved_evidence": evidence}
