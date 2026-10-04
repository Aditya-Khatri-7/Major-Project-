"""Minimal Gemini client with the same `.messages.create(...)` surface the judge tools use for Anthropic.

Talks to the Gemini REST API through httpx (no extra SDK). The judge code builds Anthropic-style content blocks
({"type": "text"} / {"type": "image", "source": {...base64...}}); this adapter converts them to Gemini `parts`.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import httpx

from config import settings

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _to_parts(content: list[dict]) -> list[dict]:
    parts: list[dict] = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"text": block["text"]})
        elif block.get("type") == "image":
            src = block["source"]
            parts.append({"inlineData": {"mimeType": src["media_type"], "data": src["data"]}})
        else:
            raise ValueError(f"unsupported content block type: {block.get('type')!r}")
    return parts


class _Messages:
    def __init__(self, owner: "GeminiClient"):
        self._owner = owner

    def create(self, *, model: str, max_tokens: int, system: str, messages: list[dict], temperature: float = 0.0, **_):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": _to_parts(m["content"])} for m in messages],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max(max_tokens, 1024),
                                 "responseMimeType": "application/json"},
        }
        data = self._owner.post(model, body)
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates (promptFeedback={data.get('promptFeedback')})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class GeminiClient:
    def __init__(self, api_key: str, max_retries: int = 5, timeout: float = 60.0, transport: httpx.BaseTransport | None = None):
        self._key = api_key
        self._retries = max_retries
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self.messages = _Messages(self)

    def post(self, model: str, body: dict) -> dict:
        url = f"{API_ROOT}/{model or settings.gemini_model_id}:generateContent"
        delay = 1.0
        for attempt in range(self._retries + 1):
            resp = self._http.post(url, json=body, headers={"x-goog-api-key": self._key})
            if resp.status_code in RETRY_STATUS and attempt < self._retries:
                time.sleep(delay)
                delay = min(delay * 2, 20.0)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
            return resp.json()
        raise RuntimeError("Gemini API: retries exhausted")
