"""API hardening: API key, rate limit, queue timeout, retention, knowledge-base bootstrap."""
import os
import time

import pytest
from fastapi.testclient import TestClient

from api import main, security
from config import settings

WORDS = " ".join(["word"] * 40)


class FakeGraph:
    def invoke(self, state):
        return {"job_id": state["job_id"], "modality": state["modality"], "final_verdict": "authentic", "final_confidence": 0.9,
                "fused_score": 0.1, "escalate_to_human": False, "report": {}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    security.reset_rate_limit()
    monkeypatch.setattr(main, "get_graph", lambda: FakeGraph())
    yield
    security.reset_rate_limit()


def test_api_key_required_only_when_configured(monkeypatch):
    client = TestClient(main.app)
    assert client.post("/analyze/text", json={"text": WORDS}).status_code == 200
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert client.post("/analyze/text", json={"text": WORDS}).status_code == 401
    assert client.post("/analyze/text", json={"text": WORDS}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/analyze/text", json={"text": WORDS}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200            # liveness stays open


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_min", 3)
    client = TestClient(main.app)
    codes = [client.post("/analyze/text", json={"text": WORDS}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    monkeypatch.setattr(settings, "rate_limit_per_min", 0)
    assert client.post("/analyze/text", json={"text": WORDS}).status_code == 200


def test_busy_queue_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "queue_timeout_s", 0)
    assert main._pipeline_slot.acquire(timeout=1)
    try:
        assert TestClient(main.app).post("/analyze/text", json={"text": WORDS}).status_code == 503
    finally:
        main._pipeline_slot.release()
    assert TestClient(main.app).post("/analyze/text", json={"text": WORDS}).status_code == 200   # slot released again


def test_purge_old_jobs_removes_only_old_folders(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for d in (old, new):
        d.mkdir()
        (d / "input.png").write_bytes(b"x")
    past = time.time() - 40 * 86400
    os.utime(old, (past, past))
    assert security.purge_old_jobs(tmp_path, days=0) == 0           # disabled
    assert security.purge_old_jobs(tmp_path, days=30) == 1
    assert not old.exists() and new.exists()


def test_ensure_knowledge_base_populates_empty_collection(tmp_path, monkeypatch, fake_embeddings):
    from rag import ingest, store
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    store.reset_clients()
    store.set_embedding_function(__import__("tests.conftest", fromlist=["HashEmbedding"]).HashEmbedding())
    assert ingest.ensure_knowledge_base() > 0
    n = store.get_kb().count()
    assert ingest.ensure_knowledge_base() == n                      # idempotent: does not re-ingest
