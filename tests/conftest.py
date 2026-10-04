"""Test configuration: isolate all storage in a temp dir and never touch real models or the network."""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="forensics_tests_"))
os.environ.update({
    "EVIDENCE_DB_URL": f"sqlite:///{(_TMP / 'evidence.db').as_posix()}",
    "CHROMA_PERSIST_DIR": str(_TMP / "chroma"),
    "JOBS_DIR": str(_TMP / "jobs"),
    "CALIBRATION_PATH": str(_TMP / "no_calibration.json"),
    "ANTHROPIC_API_KEY": "",
    "LOG_LEVEL": "WARNING",
})

import hashlib  # noqa: E402

import pytest  # noqa: E402

from agents.schemas import ToolVerdict  # noqa: E402


def make_verdict(name="text_dl", score=0.8, confidence=0.6, modality="text", error=False, **features) -> ToolVerdict:
    if error:
        return ToolVerdict.failure(name, modality, "boom")
    return ToolVerdict(tool_name=name, modality=modality, score=score, confidence=confidence,
                       explanation=f"{name} says {score}", raw_features=features)


class HashEmbedding:
    """Deterministic offline embedding for tests (bag of hashed words), so no model download is needed."""

    def __call__(self, input):
        out = []
        for text in input:
            vec = [0.0] * 64
            for w in text.lower().split():
                vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_query(self, input):          # chromadb >= 1.x calls this for query-time embeddings
        return self(input)

    def embed_documents(self, input):
        return self(input)

    @staticmethod
    def is_legacy():
        return True

    def name(self):
        return "hash-embedding-test"


@pytest.fixture
def fake_embeddings():
    from rag import store

    store.reset_clients()
    store.set_embedding_function(HashEmbedding())
    yield
    store.reset_clients()


@pytest.fixture(autouse=True)
def _isolate_from_local_env(monkeypatch):
    """Tests must not depend on a developer's .env (real API keys, provider, access key)."""
    from config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "api_key", "")
