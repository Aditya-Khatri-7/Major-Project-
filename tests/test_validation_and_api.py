import io
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agents.schemas import initial_state
from api import main
from api.validation import InputRejected, sniff_image_type, validate_image_bytes, validate_text
from config import settings
from tests.conftest import make_verdict

WORDS = " ".join(["word"] * 40)


def image_bytes(fmt="PNG", size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, format=fmt)
    return buf.getvalue()


def test_text_validation_limits():
    assert validate_text(f"  {WORDS}  ") == WORDS
    with pytest.raises(InputRejected):
        validate_text("too short")
    with pytest.raises(InputRejected) as exc:
        validate_text("w " * (settings.max_text_words + 1))
    assert exc.value.status == 413


def test_image_validation_accepts_real_images_and_rejects_fakes():
    assert validate_image_bytes(image_bytes("PNG")) == ".png"
    assert validate_image_bytes(image_bytes("JPEG")) == ".jpg"
    with pytest.raises(InputRejected) as exc:
        validate_image_bytes(b"MZ\x90\x00 not an image at all" * 10)
    assert exc.value.status == 415
    with pytest.raises(InputRejected):
        validate_image_bytes(b"")
    with pytest.raises(InputRejected):
        validate_image_bytes(image_bytes("PNG")[:40])                        # truncated file
    assert sniff_image_type(b"\x89PNG\r\n\x1a\n" + b"0" * 8) == "png"


def test_oversized_image_rejected(monkeypatch):
    monkeypatch.setattr(settings, "max_image_side_px", 32)
    with pytest.raises(InputRejected) as exc:
        validate_image_bytes(image_bytes("PNG", (64, 48)))
    assert exc.value.status == 413


class FakeGraph:
    def invoke(self, state):
        state = dict(state)
        state.update(final_verdict="synthetic", final_confidence=0.9, fused_score=0.91, escalate_to_human=False,
                     verifier_notes="ok", citations=["text_binoculars"], retries=0,
                     tool_verdicts=[make_verdict("text_dl", 0.9, 0.8)])
        state["report"] = {
            "tool_verdicts": [{"tool": "text_dl", "score": 0.9, "confidence": 0.8, "explanation": "e", "error": False,
                               "features": {"gradcam_file": "gradcam.png"} if state["modality"] == "image" else {}}],
            "evidence": [], "citations": ["text_binoculars"], "summary": "s", "reflexion_used": False,
        }
        return state


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "get_graph", lambda: FakeGraph())
    return TestClient(main.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_configured"] is False


def test_analyze_text_ok_and_validation_errors(client):
    r = client.post("/analyze/text", json={"text": WORDS})
    assert r.status_code == 200 and r.json()["verdict"] == "synthetic" and r.json()["citations"] == ["text_binoculars"]
    assert client.post("/analyze/text", json={"text": "short"}).status_code == 400
    assert client.post("/analyze/text", json={}).status_code == 422


def test_analyze_image_upload_saves_file_and_returns_gradcam_url(client):
    r = client.post("/analyze/image", files={"file": ("x.png", image_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["gradcam_url"] == f"/jobs/{body['job_id']}/gradcam"
    assert (Path(settings.jobs_dir) / body["job_id"] / "input.png").exists()


def test_analyze_image_rejects_non_images_and_ignores_client_filename(client):
    bad = client.post("/analyze/image", files={"file": ("../../evil.png", b"not an image", "image/png")})
    assert bad.status_code == 415


def test_pipeline_failure_is_a_generic_500(monkeypatch):
    class Boom:
        def invoke(self, state):
            raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main, "get_graph", lambda: Boom())
    r = TestClient(main.app).post("/analyze/text", json={"text": WORDS})
    assert r.status_code == 500 and "secret" not in r.text


def test_job_lookup_rejects_bad_ids_and_serves_records(client, fake_embeddings):
    assert client.get("/jobs/not-a-uuid").status_code == 404
    assert client.get("/jobs/../../etc/passwd/gradcam").status_code == 404
    unknown = str(uuid.uuid4())
    assert client.get(f"/jobs/{unknown}").status_code == 404
    assert client.post(f"/jobs/{unknown}/correction", json={"label": "authentic"}).status_code == 404


def test_correction_updates_record_and_case_memory(client, fake_embeddings):
    from agents.orchestrator import persist_evidence

    job_id = str(uuid.uuid4())
    state = initial_state(job_id, "text", WORDS)
    state.update(final_verdict="synthetic", final_confidence=0.8, fused_score=0.9,
                 tool_verdicts=[make_verdict("text_dl", 0.9, 0.8)], report={"model_versions": {}})
    persist_evidence(state)
    assert client.get(f"/jobs/{job_id}").json()["final_verdict"] == "synthetic"
    r = client.post(f"/jobs/{job_id}/correction", json={"label": "authentic"})
    assert r.status_code == 200 and r.json()["case_memory_updated"] is True
    assert client.get(f"/jobs/{job_id}").json()["human_correction"] == "authentic"
    assert client.post(f"/jobs/{job_id}/correction", json={"label": "maybe"}).status_code == 422
