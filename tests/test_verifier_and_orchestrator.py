import pytest

from agents import orchestrator, reflexion, verifier_agent
from agents.report import build_report
from agents.schemas import RetrievedEvidence, initial_state
from config import settings
from tests.conftest import make_verdict

TEXT = " ".join(["word"] * 60)


def state_with(verdicts, evidence=None, **kw):
    s = initial_state("job-1", "text", TEXT)
    s["tool_verdicts"] = verdicts
    s["retrieved_evidence"] = evidence or []
    s.update(kw)
    return s


def test_verifier_writes_all_decision_fields():
    out = verifier_agent.run_verifier(state_with([make_verdict("text_dl", 0.9, 0.8), make_verdict("text_llm", 0.85, 0.7)]))
    assert out["final_verdict"] == "synthetic" and out["fused_score"] > 0.8
    assert out["escalate_to_human"] is False and out["needs_reflexion"] is False


def test_reflexion_needed_only_when_a_working_judge_and_api_key_exist(monkeypatch):
    disagree = [make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.1, 0.9)]
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert verifier_agent.run_verifier(state_with(disagree))["needs_reflexion"] is False
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    assert verifier_agent.run_verifier(state_with(disagree))["needs_reflexion"] is True
    assert verifier_agent.run_verifier(state_with(disagree, retries=1))["needs_reflexion"] is False


def test_contradicting_verified_case_forces_escalation():
    ev = [RetrievedEvidence(source_id="case:abc", content_snippet="x", similarity_score=0.9,
                            evidence_type="case_study", verified=True, label="authentic")]
    out = verifier_agent.run_verifier(state_with([make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.9, 0.9)], ev))
    assert out["final_verdict"] == "synthetic" and out["escalate_to_human"] is True
    assert "verified case" in out["verifier_notes"]


def test_unverified_or_distant_neighbours_do_not_matter():
    ev = [RetrievedEvidence(source_id="case:abc", content_snippet="x", similarity_score=0.6,
                            evidence_type="case_study", verified=True, label="authentic"),
          RetrievedEvidence(source_id="case:def", content_snippet="x", similarity_score=0.95,
                            evidence_type="case_study", verified=False, label="authentic")]
    out = verifier_agent.run_verifier(state_with([make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.9, 0.9)], ev))
    assert out["escalate_to_human"] is False


def test_report_is_citation_forced():
    s = state_with([make_verdict("text_dl", 0.9, 0.8)])
    s.update(verifier_agent.run_verifier(s))
    none_case = build_report(s)
    assert none_case["citations"] == [] and "No supporting evidence" in none_case["report"]["summary"]
    s["retrieved_evidence"] = [RetrievedEvidence(source_id="text_binoculars", content_snippet="n", similarity_score=0.7,
                                                 evidence_type="literature")]
    with_case = build_report(s)
    assert with_case["citations"] == ["text_binoculars"] and "text_binoculars" in with_case["report"]["summary"]


def test_reflexion_replaces_judge_verdict(monkeypatch):
    from agents.text_agent.tools import llm_judge

    calls = {}

    def fake(text, peers, evidence, client=None):
        calls["peers"] = peers
        return make_verdict("text_llm", 0.7, 0.8, reflexion=True)

    monkeypatch.setattr(llm_judge, "score_text_with_context", fake)
    verdicts = [make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.1, 0.9)]
    out = reflexion.rerun_judge("text", TEXT, verdicts, [])
    judge = next(v for v in out if v.tool_name == "text_llm")
    assert judge.score == 0.7 and judge.raw_features["previous_score"] == 0.1
    assert "text_dl" in calls["peers"] and "text_llm" not in calls["peers"]


def test_reflexion_keeps_old_verdict_when_rerun_fails(monkeypatch):
    from agents.text_agent.tools import llm_judge

    monkeypatch.setattr(llm_judge, "score_text_with_context",
                        lambda *a, **k: make_verdict("text_llm", error=True))
    verdicts = [make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.1, 0.9)]
    out = reflexion.rerun_judge("text", TEXT, verdicts, [])
    assert next(v for v in out if v.tool_name == "text_llm").score == 0.1


@pytest.fixture
def stub_graph(monkeypatch, fake_embeddings):
    """Full graph with fake tools: disagreement on the first pass, agreement after Reflexion."""
    calls = {"judge": 0}

    def text_agent(state):
        return {"tool_verdicts": [make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.1, 0.9)]}

    def rerun(modality, input_ref, verdicts, evidence, client=None):
        calls["judge"] += 1
        return [v if v.tool_name != "text_llm" else make_verdict("text_llm", 0.9, 0.8) for v in verdicts]

    monkeypatch.setattr(orchestrator, "run_text_agent", text_agent)
    monkeypatch.setattr(reflexion, "rerun_judge", rerun)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "enable_reflexion", True)
    return orchestrator.build_graph(), calls


def test_graph_runs_reflexion_once_then_reports_and_persists(stub_graph):
    graph, calls = stub_graph
    out = graph.invoke(initial_state("11111111-1111-4111-8111-111111111111", "text", TEXT))
    assert calls["judge"] == 1 and out["retries"] == 1
    assert out["final_verdict"] == "synthetic" and out["report"]["reflexion_used"] is True
    assert out["report"]["final_verdict"] == "synthetic"
    from evidence_store.models import get_record

    record = get_record("11111111-1111-4111-8111-111111111111")
    assert record["final_verdict"] == "synthetic" and len(record["tool_verdicts"]) == 2
    assert len(record["input_hash"]) == 64 and "word word" not in str(record["input_meta"])


def _image_state(*verdicts):
    s = initial_state("job-img", "image", "x.png")
    s["tool_verdicts"] = list(verdicts)
    return s


def test_sole_general_probe_decisive_verdict_is_not_force_escalated():
    out = verifier_agent.run_verifier(_image_state(make_verdict("image_general", 0.97, 0.9, modality="image")))
    assert out["final_verdict"] == "synthetic" and out["escalate_to_human"] is False
    assert out["final_confidence"] <= 0.8 and "designated detector" in out["verifier_notes"]


def test_sole_general_probe_uncertain_or_weak_verdict_still_escalates():
    out = verifier_agent.run_verifier(_image_state(make_verdict("image_general", 0.5, 0.9, modality="image")))
    assert out["escalate_to_human"] is True
    # a lone face-tool verdict is NOT a designated sole detector: still escalates
    out = verifier_agent.run_verifier(_image_state(make_verdict("image_dl", 0.99, 0.9, modality="image")))
    assert out["escalate_to_human"] is True
