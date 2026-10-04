"""FastAPI layer: validation, job bookkeeping and HTTP mapping only. No detection logic lives here."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agents.common import log
from agents.schemas import initial_state
from api.schemas import (AnalyzeResponse, CorrectionRequest, CorrectionResponse, EvidenceOut, TextRequest,
                         ToolVerdictOut)
from api.validation import InputRejected, validate_image_bytes, validate_text
from config import settings
from observability import log_event

app = FastAPI(title="Forensics Agent API", version="2.0")

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_pipeline_slot = threading.Semaphore(1)        # one job at a time: models share a single small GPU


def get_graph():
    from agents.orchestrator import get_graph as build

    return build()


def _job_id_or_404(job_id: str) -> str:
    if not _UUID.match(job_id):
        raise HTTPException(404, "Unknown job.")
    return job_id


def _run(state: dict) -> dict:
    try:
        with _pipeline_slot:
            return get_graph().invoke(state)
    except Exception as exc:
        log_event(log, "pipeline_failed", level=logging.ERROR, job_id=state["job_id"], error=str(exc))
        raise HTTPException(500, "Analysis failed. See server logs for the job id.") from exc


def _to_response(result: dict) -> AnalyzeResponse:
    report = result.get("report") or {}
    gradcam_url = None
    for v in report.get("tool_verdicts", []):
        if v.get("features", {}).get("gradcam_file"):
            gradcam_url = f"/jobs/{result['job_id']}/gradcam"
    return AnalyzeResponse(
        job_id=result["job_id"],
        modality=result["modality"],
        verdict=result.get("final_verdict"),
        confidence=result.get("final_confidence"),
        fused_score=result.get("fused_score"),
        escalate_to_human=result.get("escalate_to_human", False),
        reflexion_used=report.get("reflexion_used", False),
        verifier_notes=result.get("verifier_notes"),
        tool_verdicts=[
            ToolVerdictOut(tool=v["tool"], score=v["score"], confidence=v["confidence"],
                           explanation=v["explanation"], error=v.get("error", False), features=v.get("features", {}))
            for v in report.get("tool_verdicts", [])
        ],
        evidence=[EvidenceOut(**e) for e in report.get("evidence", [])],
        citations=report.get("citations", []),
        summary=report.get("summary"),
        gradcam_url=gradcam_url,
    )


@app.post("/analyze/text", response_model=AnalyzeResponse)
def analyze_text(req: TextRequest):
    try:
        text = validate_text(req.text)
    except InputRejected as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    state = initial_state(str(uuid.uuid4()), "text", text)
    return _to_response(_run(state))


@app.post("/analyze/image", response_model=AnalyzeResponse)
def analyze_image(file: UploadFile = File(...)):
    limit = settings.max_image_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    try:
        extension = validate_image_bytes(data)
    except InputRejected as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    job_id = str(uuid.uuid4())
    job_dir = Path(settings.jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"input{extension}"
    path.write_bytes(data)
    return _to_response(_run(initial_state(job_id, "image", str(path))))


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    from evidence_store.models import get_record

    record = get_record(_job_id_or_404(job_id))
    if record is None:
        raise HTTPException(404, "Unknown job.")
    return record


@app.get("/jobs/{job_id}/gradcam")
def get_gradcam(job_id: str):
    path = Path(settings.jobs_dir) / _job_id_or_404(job_id) / "gradcam.png"
    if not path.is_file():
        raise HTTPException(404, "No Grad-CAM image for this job.")
    return FileResponse(path, media_type="image/png")


@app.post("/jobs/{job_id}/correction", response_model=CorrectionResponse)
def correct_job(job_id: str, body: CorrectionRequest):
    from evidence_store.models import set_correction
    from rag.case_memory import apply_correction

    _job_id_or_404(job_id)
    if not set_correction(job_id, body.label):
        raise HTTPException(404, "Unknown job.")
    try:
        updated = apply_correction(job_id, body.label)
    except Exception as exc:
        log_event(log, "case_memory_update_failed", level=logging.WARNING, job_id=job_id, error=str(exc))
        updated = False
    return CorrectionResponse(job_id=job_id, label=body.label, case_memory_updated=updated)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(settings.anthropic_api_key),
        "text_dl_model": Path(settings.text_dl_model_path, "config.json").exists(),
        "image_dl_model": Path(settings.image_model_path).exists(),
        "calibration_file": Path(settings.calibration_path).exists(),
    }
