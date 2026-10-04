"""Scope routing: face photos use the face tools; non-face images use only the general AI-image probe."""
from agents.image_agent import agent
from agents.schemas import ToolVerdict


def _fake_runner(name):
    return lambda path, **_: ToolVerdict(tool_name=name, modality="image", score=0.5, confidence=0.5, explanation="stub")


def _run(monkeypatch, face_p):
    monkeypatch.setattr(agent, "_face_probability", lambda path: face_p)
    monkeypatch.setattr(agent, "_runners", lambda: {k: _fake_runner(f"image_{k}") for k in ("dl", "clip", "general", "vlm")})
    monkeypatch.setattr(agent.settings, "enabled_image_tools", "dl,clip,general")
    out = agent.run_image_agent({"input_ref": "x.png"})["tool_verdicts"]
    return {v.tool_name: v.error for v in out}


def test_face_image_uses_face_tools_only(monkeypatch):
    res = _run(monkeypatch, 0.9)
    assert res == {"image_dl": False, "image_clip": False}


def test_non_face_image_uses_general_probe_only(monkeypatch):
    res = _run(monkeypatch, 0.1)
    assert res["image_general"] is False
    assert res["image_dl"] is True and res["image_clip"] is True      # skipped with an explanation, not scored


def test_scope_check_failure_falls_back_to_face_tools(monkeypatch):
    res = _run(monkeypatch, None)
    assert res == {"image_dl": False, "image_clip": False}
