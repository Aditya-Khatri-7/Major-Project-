from agents.calibration import DEFAULTS
from agents.fusion import fuse, length_factor
from tests.conftest import make_verdict


def cfg(**overrides):
    import copy
    c = copy.deepcopy(DEFAULTS)
    c.update(overrides)
    return c


def test_agreeing_tools_give_confident_verdict_without_escalation():
    v = [make_verdict("text_dl", 0.92, 0.84), make_verdict("text_llm", 0.88, 0.76)]
    r = fuse(v, "text", 200, cfg())
    assert r.verdict == "synthetic" and not r.escalate and not r.needs_reflexion
    assert 0.85 < r.fused < 0.95


def test_error_and_low_confidence_tools_are_excluded():
    v = [make_verdict("text_dl", 0.9, 0.8), make_verdict("text_slm", error=True),
         make_verdict("text_llm", 0.1, 0.05)]
    r = fuse(v, "text", 200, cfg())
    assert r.usable == ["text_dl"]
    assert r.escalate                                   # single usable tool -> escalate
    assert not r.needs_reflexion


def test_no_usable_tools_is_uncertain_and_escalated():
    r = fuse([make_verdict("text_dl", error=True)], "text", 100, cfg())
    assert r.fused is None and r.verdict == "uncertain" and r.escalate and r.confidence == 0.0


def test_disagreement_triggers_reflexion_once_and_lowers_confidence():
    v = [make_verdict("text_dl", 0.95, 0.9), make_verdict("text_llm", 0.10, 0.9)]
    r = fuse(v, "text", 200, cfg(), retries=0)
    assert r.spread > 0.35 and r.needs_reflexion and r.escalate
    r2 = fuse(v, "text", 200, cfg(), retries=1)
    assert not r2.needs_reflexion and r2.escalate       # still escalated, but no further retry


def test_uncertain_band_between_thresholds():
    v = [make_verdict("text_dl", 0.5, 0.5), make_verdict("text_llm", 0.55, 0.5)]
    r = fuse(v, "text", 200, cfg())
    assert r.verdict == "uncertain" and r.escalate


def test_thresholds_are_read_from_config():
    v = [make_verdict("image_dl", 0.6, 0.5, modality="image"), make_verdict("image_vlm", 0.62, 0.5, modality="image")]
    c = cfg()
    c["thresholds"]["image"] = {"t_lo": 0.3, "t_hi": 0.55, "t_star": 0.5}
    assert fuse(v, "image", 0, c).verdict == "synthetic"


def test_short_text_downweights_slm_and_dl():
    assert length_factor("text_slm", 60) == 0.5 and length_factor("text_slm", 150) == 1.0
    assert length_factor("text_dl", 30) == 0.7 and length_factor("text_llm", 30) == 1.0
    short = [make_verdict("text_slm", 0.9, 0.8), make_verdict("text_llm", 0.4, 0.8)]
    weights_short = fuse(short, "text", 60, cfg()).weights
    weights_long = fuse(short, "text", 300, cfg()).weights
    assert weights_short["text_slm"] < weights_long["text_slm"]


def test_learned_weights_shift_fused_score():
    v = [make_verdict("text_dl", 0.9, 0.5), make_verdict("text_llm", 0.6, 0.5)]
    c = cfg()
    c["weights"] = {"text_dl": 1.0, "text_llm": 0.05}
    heavy = fuse(v, "text", 200, c).fused
    c["weights"] = {"text_dl": 0.05, "text_llm": 1.0}
    light = fuse(v, "text", 200, c).fused
    assert heavy > light
