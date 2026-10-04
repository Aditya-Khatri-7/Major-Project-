from typing import Literal, Optional

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., description="Text to analyse (20 to 20,000 words).")


class ToolVerdictOut(BaseModel):
    tool: str
    score: float
    confidence: float
    explanation: str
    error: bool = False
    features: dict = Field(default_factory=dict)


class EvidenceOut(BaseModel):
    source_id: str
    content_snippet: str
    similarity_score: float
    evidence_type: str
    verified: bool = False
    label: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str
    modality: Literal["text", "image"]
    verdict: Optional[Literal["authentic", "synthetic", "uncertain"]]
    confidence: Optional[float]
    fused_score: Optional[float]
    escalate_to_human: bool
    reflexion_used: bool = False
    verifier_notes: Optional[str] = None
    tool_verdicts: list[ToolVerdictOut]
    evidence: list[EvidenceOut] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    gradcam_url: Optional[str] = None


class CorrectionRequest(BaseModel):
    label: Literal["authentic", "synthetic"]


class CorrectionResponse(BaseModel):
    job_id: str
    label: str
    case_memory_updated: bool
