"""Gemini adapter and provider switch. No network: httpx.MockTransport."""
import json

import httpx
import pytest

from agents.gemini_client import GeminiClient
from agents.llm_utils import parse_judge_output
from config import settings

MSG = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]


def test_converts_blocks_and_returns_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-goog-api-key"]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"score": 0.8, "confidence": 0.6, "reason": "x"}'}]}}]})

    client = GeminiClient("k", transport=httpx.MockTransport(handler))
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}, {"type": "text", "text": "hi"}]
    resp = client.messages.create(model="gemini-x", max_tokens=100, system="sys", messages=[{"role": "user", "content": content}])
    assert seen["key"] == "k"
    assert seen["body"]["contents"][0]["parts"][0] == {"inlineData": {"mimeType": "image/png", "data": "QUJD"}}
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert parse_judge_output(resp.content[0].text).score == 0.8


def test_http_error_and_no_candidates_raise():
    bad = GeminiClient("k", max_retries=0, transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad key")))
    with pytest.raises(RuntimeError, match="400"):
        bad.messages.create(model="m", max_tokens=10, system="s", messages=MSG)
    blocked = GeminiClient("k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})))
    with pytest.raises(RuntimeError, match="no candidates"):
        blocked.messages.create(model="m", max_tokens=10, system="s", messages=MSG)


def test_retries_transient_status(monkeypatch):
    monkeypatch.setattr("agents.gemini_client.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    GeminiClient("k", transport=httpx.MockTransport(handler)).messages.create(model="m", max_tokens=10, system="s", messages=MSG)
    assert calls["n"] == 3


def test_provider_switch_selects_key_and_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "g")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert settings.llm_configured and settings.judge_model_id == settings.gemini_model_id
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    assert not settings.llm_configured
