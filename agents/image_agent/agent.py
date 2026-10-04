"""Image agent: runs the enabled image tools and returns their verdicts."""
from __future__ import annotations

from agents.common import IMAGE_TOOL_NAMES, log, run_tool
from agents.schemas import ForensicState, ToolVerdict
from config import settings
from observability import log_event


def _runners() -> dict:
    from agents.image_agent.tools import clip_probe, dl_classifier, general_probe, vlm_judge

    return {"dl": dl_classifier.score_image, "clip": clip_probe.score_image, "general": general_probe.score_image, "vlm": vlm_judge.score_image}


SCOPE_THRESHOLD = 0.5
FACE_TOOLS = {"dl", "clip"}           # trained only on face photos / crops


def _face_probability(path: str):
    try:
        from agents.image_agent.scope import face_probability
        return face_probability(path)
    except Exception as exc:                              # guard must never break the pipeline
        log_event(log, "scope_check_failed", error=f"{type(exc).__name__}: {exc}")
        return None


def run_image_agent(state: ForensicState) -> dict:
    path = state["input_ref"]
    runners = _runners()
    verdicts = []
    face_p = _face_probability(path)
    out_of_scope = face_p is not None and face_p < SCOPE_THRESHOLD
    if out_of_scope:
        log_event(log, "image_out_of_scope", face_probability=round(face_p, 3))
    for key in settings.image_tools:
        if key == "general" and not out_of_scope:
            continue                      # face photos are judged by the face-deepfake tools only
        if out_of_scope and key in FACE_TOOLS:
            verdicts.append(ToolVerdict.failure(
                IMAGE_TOOL_NAMES[key], "image",
                f"Not used: no face photo detected (face probability {face_p:.2f}). This detector is trained on face "
                "deepfakes; the general AI-image probe judges non-face images instead."))
            continue
        fn = runners.get(key)
        if fn is None:
            log_event(log, "unknown_image_tool", tool=key)
            continue
        verdicts.append(run_tool(IMAGE_TOOL_NAMES[key], "image", fn, path))
    return {"tool_verdicts": verdicts}
