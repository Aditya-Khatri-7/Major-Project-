from agents.schemas import initial_state
from rag import case_memory, ingest, retriever
from rag.store import get_cases, get_kb
from tests.conftest import make_verdict

NOTE = """---
modality: text
---
# Binoculars

Binoculars divides log perplexity by cross perplexity using an observer and a performer model.

# Limits

Short passages give unreliable Binoculars scores.
"""
IMAGE_NOTE = """---
modality: image
---
# Diffusion

Diffusion images show smooth skin and garbled text on signs.
"""


def build_kb(tmp_path):
    kb = tmp_path / "kb"
    (kb / "literature").mkdir(parents=True)
    (kb / "fingerprints").mkdir()
    (kb / "literature" / "text_binoculars.md").write_text(NOTE, encoding="utf-8")
    (kb / "fingerprints" / "image_diffusion_fingerprint.md").write_text(IMAGE_NOTE, encoding="utf-8")
    (kb / "README.md").write_text("# ignored", encoding="utf-8")
    return kb


def test_front_matter_and_chunking():
    meta, body = ingest.parse_front_matter(NOTE)
    assert meta == {"modality": "text"} and body.startswith("# Binoculars")
    chunks = ingest.chunk_text(body)
    assert len(chunks) == 2 and chunks[0].startswith("# Binoculars")
    long = "# H\n" + " ".join(f"w{i}" for i in range(600))
    assert len(ingest.chunk_text(long, 220, 40)) >= 3


def test_evidence_type_from_folder(tmp_path):
    from pathlib import Path
    assert ingest.evidence_type(Path("kb/fingerprints/x.md")) == "generator_fingerprint"
    assert ingest.evidence_type(Path("kb/literature/x.md")) == "literature"


def test_ingest_is_idempotent_and_skips_readme(tmp_path, fake_embeddings):
    kb = build_kb(tmp_path)
    total = ingest.ingest_dir(kb)
    assert total == 3 and get_kb().count() == 3
    ingest.ingest_dir(kb)
    assert get_kb().count() == 3                                       # re-ingest replaces, never duplicates


def test_retrieval_filters_by_modality_and_similarity(tmp_path, fake_embeddings):
    ingest.ingest_dir(build_kb(tmp_path))
    hits = retriever.retrieve_evidence("Binoculars divides log perplexity by cross perplexity observer performer", "text")
    assert hits and hits[0].source_id == "text_binoculars" and hits[0].evidence_type == "literature"
    assert all(h.similarity_score >= retriever.MIN_SIMILARITY for h in hits)
    assert all(h.source_id != "image_diffusion_fingerprint" for h in hits)          # image note filtered out for text
    assert retriever.retrieve_evidence("zzz qqq unrelated tokens entirely", "text") == []


def test_rag_node_uses_rewrite_and_never_raises(tmp_path, fake_embeddings, monkeypatch):
    ingest.ingest_dir(build_kb(tmp_path))
    state = initial_state("j", "text", "x")
    state["tool_verdicts"] = [make_verdict("text_slm", 0.8, 0.7)]
    out = retriever.run_rag_node(state)
    assert isinstance(out["retrieved_evidence"], list)
    state["tool_verdicts"] = [make_verdict("text_dl", error=True)]
    assert retriever.run_rag_node(state) == {"retrieved_evidence": []}
    monkeypatch.setattr(retriever, "get_kb", lambda: (_ for _ in ()).throw(RuntimeError("chroma down")))
    state["tool_verdicts"] = [make_verdict("text_dl", 0.9, 0.9)]
    assert retriever.run_rag_node(state) == {"retrieved_evidence": []}


def test_case_memory_roundtrip_with_human_correction(fake_embeddings):
    state = initial_state("job-a", "text", "x")
    state.update(final_verdict="synthetic", fused_score=0.9, tool_verdicts=[make_verdict("text_dl", 0.9, 0.8)])
    case_memory.add_case(state)
    assert get_cases().count() == 1
    assert case_memory.apply_correction("nope", "authentic") is False
    assert case_memory.apply_correction("job-a", "authentic") is True
    hits = retriever.retrieve_evidence(case_memory.case_document(state), "text", kb_n=0)
    case = next(h for h in hits if h.source_id == "case:job-a")
    assert case.verified is True and case.label == "authentic" and case.evidence_type == "case_study"
    assert retriever.retrieve_evidence(case_memory.case_document(state), "image", kb_n=0) == []      # modality-scoped
