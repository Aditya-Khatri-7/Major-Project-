"""Shared state schema for the forensics pipeline (single source of truth).

Every node in the LangGraph orchestrator reads from and writes to ForensicState.
Rule: every key a node writes MUST be declared here, otherwise LangGraph drops it.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Modality = Literal["text", "image"]
Verdict = Literal["authentic", "synthetic", "uncertain"]


class ToolVerdict(BaseModel):
    tool_name: str
    modality: Modality
    score: float = Field(ge=0.0, le=1.0)        # calibrated P(AI / fake): 0 = authentic, 1 = synthetic
    confidence: float = Field(ge=0.0, le=1.0)   # how much to trust this tool on THIS input
    explanation: str
    raw_features: dict = Field(default_factory=dict)
    error: bool = False                         # True => excluded from fusion and disagreement

    @classmethod
    def failure(cls, tool_name: str, modality: Modality, reason: str) -> "ToolVerdict":
        return cls(
            tool_name=tool_name, modality=modality, score=0.5, confidence=0.0,
            explanation=reason[:300], error=True,
        )


class RetrievedEvidence(BaseModel):
    source_id: str
    content_snippet: str
    similarity_score: float                     # cosine similarity, 0..1
    evidence_type: Literal["generator_fingerprint", "case_study", "literature"]
    verified: bool = False                      # case memory only: confirmed by a human
    label: Optional[str] = None                 # case memory only: verdict / corrected label


class ForensicState(TypedDict):
    job_id: str
    modality: Modality
    input_ref: str                              # raw text, or path to the uploaded image
    tool_verdicts: list[ToolVerdict]
    retrieved_evidence: list[RetrievedEvidence]
    retries: int                                # Reflexion counter (max 1)
    needs_reflexion: bool
    fused_score: Optional[float]                # P(AI) after fusion; benchmarks and plots use THIS
    verifier_notes: Optional[str]
    final_verdict: Optional[Verdict]
    final_confidence: Optional[float]
    escalate_to_human: bool
    citations: list[str]                        # source ids actually retrieved and cited
    report: Optional[dict]


def initial_state(job_id: str, modality: Modality, input_ref: str) -> ForensicState:
    return {
        "job_id": job_id, "modality": modality, "input_ref": input_ref,
        "tool_verdicts": [], "retrieved_evidence": [],
        "retries": 0, "needs_reflexion": False,
        "fused_score": None, "verifier_notes": None,
        "final_verdict": None, "final_confidence": None,
        "escalate_to_human": False, "citations": [], "report": None,
    }
