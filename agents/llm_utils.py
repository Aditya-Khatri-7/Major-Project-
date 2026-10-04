"""Shared helpers for the LLM / VLM judge tools: robust JSON parsing, retries, error verdicts."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field

from agents.schemas import Modality, ToolVerdict
from config import require_api_key, settings
from observability import get_logger, log_event

log = get_logger("llm")


class JudgeOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    artifacts_found: list[str] = Field(default_factory=list)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    raise ValueError("no JSON object found in model output")


def _clamp01(value) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_judge_output(raw_text: str) -> JudgeOutput:
    data = extract_json(raw_text)
    data["score"] = _clamp01(data["score"])
    data["confidence"] = _clamp01(data.get("confidence", 0.5))
    data["reason"] = str(data.get("reason") or data.get("explanation") or "")
    artifacts = data.get("artifacts_found") or []
    data["artifacts_found"] = [str(a) for a in artifacts] if isinstance(artifacts, list) else []
    return JudgeOutput(**data)


@lru_cache(maxsize=1)
def get_client():
    import anthropic
    return anthropic.Anthropic(api_key=require_api_key(), max_retries=settings.llm_max_retries)


def call_judge(system: str, content: list[dict], client=None, max_tokens: int = 500, attempts: int = 3) -> JudgeOutput:
    """Call the judge model at temperature 0 and parse its JSON reply.

    Retries on unparsable output. If the model rejects the `temperature` parameter it is dropped.
    """
    client = client or get_client()
    kwargs: dict = {"temperature": 0}
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        sys_prompt = system if attempt == 0 else system + "\nIMPORTANT: reply with one raw JSON object and nothing else."
        try:
            response = client.messages.create(
                model=settings.llm_model_id, max_tokens=max_tokens, system=sys_prompt,
                messages=[{"role": "user", "content": content}], **kwargs,
            )
            raw = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
            return parse_judge_output(raw)
        except ValueError as exc:                     # includes pydantic ValidationError
            last_error = exc
        except Exception as exc:
            if "temperature" in kwargs and "temperature" in str(exc).lower():
                kwargs.pop("temperature")
                last_error = exc
                continue
            raise
    raise ValueError(f"judge output could not be parsed after {attempts} attempts: {last_error}")


def judge_to_verdict(tool_name: str, modality: Modality, system: str, content: list[dict],
                     client=None, extra_features: Optional[dict] = None) -> ToolVerdict:
    """Run a judge call and convert the result to a ToolVerdict. Never raises."""
    try:
        out = call_judge(system, content, client=client)
    except Exception as exc:
        log_event(log, "judge_failed", tool=tool_name, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return ToolVerdict.failure(tool_name, modality, f"{type(exc).__name__}: {str(exc)[:200]}")
    features = {"artifacts_found": out.artifacts_found}
    features.update(extra_features or {})
    return ToolVerdict(
        tool_name=tool_name, modality=modality, score=out.score, confidence=out.confidence,
        explanation=out.reason or "No explanation provided.", raw_features=features,
    )
