"""ChromaDB access: one persistent client, cosine collections, one embedding function.

Two collections:
  forensics_kb  - layers 1 and 2 (literature notes, generator fingerprints), built by rag/ingest.py
  case_memory   - layer 3, one entry per analysed job, updated when a human corrects a verdict
"""
from __future__ import annotations

import threading

from config import settings

KB_COLLECTION = "forensics_kb"
CASE_COLLECTION = "case_memory"

_lock = threading.Lock()
_client = None
_embedding_fn = None


def set_embedding_function(fn) -> None:
    """Override the embedding function (used by tests to avoid downloading a model)."""
    global _embedding_fn
    _embedding_fn = fn


def get_embedding_function():
    global _embedding_fn
    with _lock:
        if _embedding_fn is None:
            from chromadb.utils import embedding_functions

            name = settings.embedding_model.replace("sentence-transformers/", "")
            _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=name, device="cpu")
        return _embedding_fn


def get_client():
    global _client
    with _lock:
        if _client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            _client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
            )
        return _client


def _collection(name: str):
    return get_client().get_or_create_collection(
        name, embedding_function=get_embedding_function(), metadata={"hnsw:space": "cosine"}
    )


def get_kb():
    return _collection(KB_COLLECTION)


def get_cases():
    return _collection(CASE_COLLECTION)


def reset_clients() -> None:
    """Forget cached client / embedding function (tests)."""
    global _client, _embedding_fn
    with _lock:
        _client = None
        _embedding_fn = None
