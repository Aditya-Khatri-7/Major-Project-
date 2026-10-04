"""Evidence store: one audit record per job (SQLite by default; use a Postgres URL in production).

Stores the input hash (never the raw content), every tool verdict, the fused score, the verdict,
the citations, and the model versions that produced them, plus any later human correction.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, String, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from config import settings


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    input_hash: Mapped[str] = mapped_column(String, nullable=False)          # SHA-256 of the input
    modality: Mapped[str] = mapped_column(String, nullable=False)
    input_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # word count / file name, no content
    final_verdict: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    final_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fused_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    escalate: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_verdicts: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    citations: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    verifier_notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_versions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    report: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    human_correction: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    corrected_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


_init_lock = threading.Lock()


@lru_cache(maxsize=4)
def _engine(url: str):
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database and parsed.database != ":memory:":
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if parsed.get_backend_name() == "sqlite" else {}
    engine = create_engine(url, connect_args=connect_args)
    with _init_lock:
        Base.metadata.create_all(engine)
    return engine


def get_engine():
    return _engine(settings.evidence_db_url)


def _clean(value):
    """Make a structure JSON-safe (numpy scalars, paths, ...)."""
    return json.loads(json.dumps(value, default=str))


def _input_meta(state: dict) -> dict:
    if state["modality"] == "text":
        return {"n_words": len(state["input_ref"].split())}
    path = Path(state["input_ref"])
    return {"file": path.name, "bytes": path.stat().st_size if path.exists() else None}


def save_record(state: dict) -> None:
    from evidence_store.hashing import hash_input

    verdicts = [v.model_dump() for v in state.get("tool_verdicts", [])]
    report = state.get("report") or {}
    record = EvidenceRecord(
        job_id=state["job_id"],
        input_hash=hash_input(state["input_ref"], state["modality"]),
        modality=state["modality"],
        input_meta=_input_meta(state),
        final_verdict=state.get("final_verdict"),
        final_confidence=state.get("final_confidence"),
        fused_score=state.get("fused_score"),
        escalate=bool(state.get("escalate_to_human", False)),
        tool_verdicts=_clean(verdicts),
        citations=_clean(state.get("citations", [])),
        verifier_notes=state.get("verifier_notes") or "",
        model_versions=_clean(report.get("model_versions", {})),
        report=_clean(report),
    )
    with Session(get_engine()) as session:
        session.merge(record)
        session.commit()


def _to_dict(rec: EvidenceRecord) -> dict:
    return {
        "job_id": rec.job_id, "input_hash": rec.input_hash, "modality": rec.modality,
        "input_meta": rec.input_meta, "final_verdict": rec.final_verdict,
        "final_confidence": rec.final_confidence, "fused_score": rec.fused_score,
        "escalate": rec.escalate, "tool_verdicts": rec.tool_verdicts, "citations": rec.citations,
        "verifier_notes": rec.verifier_notes, "model_versions": rec.model_versions, "report": rec.report,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        "human_correction": rec.human_correction,
        "corrected_at": rec.corrected_at.isoformat() if rec.corrected_at else None,
    }


def get_record(job_id: str) -> Optional[dict]:
    with Session(get_engine()) as session:
        rec = session.get(EvidenceRecord, job_id)
        return _to_dict(rec) if rec else None


def set_correction(job_id: str, label: str) -> bool:
    with Session(get_engine()) as session:
        rec = session.get(EvidenceRecord, job_id)
        if rec is None:
            return False
        rec.human_correction = label
        rec.corrected_at = _now()
        session.commit()
        return True
