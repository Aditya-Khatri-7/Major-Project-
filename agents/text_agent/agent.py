"""Text agent: runs the enabled text tools and returns their verdicts."""
from __future__ import annotations

from agents.common import TEXT_TOOL_NAMES, log, run_tool
from agents.schemas import ForensicState
from config import settings
from observability import log_event


def _runners() -> dict:
    # Imported lazily so that importing the graph does not import torch / transformers.
    from agents.text_agent.tools import dl_classifier, llm_judge, slm_binoculars

    return {"dl": dl_classifier.score_text, "slm": slm_binoculars.score_text, "llm": llm_judge.score_text}


def run_text_agent(state: ForensicState) -> dict:
    text = state["input_ref"]
    runners = _runners()
    verdicts = []
    for key in settings.text_tools:
        fn = runners.get(key)
        if fn is None:
            log_event(log, "unknown_text_tool", tool=key)
            continue
        verdicts.append(run_tool(TEXT_TOOL_NAMES[key], "text", fn, text))
    return {"tool_verdicts": verdicts}
