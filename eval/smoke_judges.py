"""One-command check that the LLM / VLM judge tools work with the configured provider and key.

    python eval/smoke_judges.py            # uses LLM_PROVIDER / GEMINI_API_KEY / ANTHROPIC_API_KEY from .env

Runs the text judge on the two demo texts and the vision judge on the two demo images and prints each verdict.
Exit code 0 only if every call returned a valid (non-error) verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.image_agent.tools import vlm_judge  # noqa: E402
from agents.text_agent.tools import llm_judge  # noqa: E402
from config import settings  # noqa: E402

DEMO = Path("demo_samples")


def main() -> int:
    if not settings.llm_configured:
        print(f"No key for provider '{settings.llm_provider}'. Set GEMINI_API_KEY (LLM_PROVIDER=gemini) or ANTHROPIC_API_KEY in .env.")
        return 2
    print(f"Provider: {settings.llm_provider}  model: {settings.judge_model_id}")
    checks = [(f"text  {p.name}", lambda p=p: llm_judge.score_text(p.read_text(encoding="utf-8"))) for p in sorted(DEMO.glob("text_*.txt"))]
    checks += [(f"image {p.name}", lambda p=p: vlm_judge.score_image(str(p))) for p in sorted(DEMO.glob("image_*.jpg"))]
    failures = 0
    for name, fn in checks:
        v = fn()
        status = "ERROR" if v.error else "ok"
        failures += v.error
        print(f"[{status}] {name:34s} score={v.score:.2f} confidence={v.confidence:.2f}  {v.explanation[:110]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
