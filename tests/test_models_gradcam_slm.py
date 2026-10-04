"""Tests that exercise the torch code paths with tiny random models (no downloads, no checkpoints)."""
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from agents.image_agent.tools import dl_classifier as image_dl  # noqa: E402
from agents.image_agent.tools.gradcam import generate_gradcam, save_overlay  # noqa: E402


def tiny_classifier():
    from torchvision import models
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    return m.eval()


def test_gradcam_shape_range_and_overlay(tmp_path):
    model = tiny_classifier()
    x = torch.randn(1, 3, 64, 64)
    cam = generate_gradcam(model, x, target_index=0)
    assert cam.shape == (64, 64) and cam.min() >= 0.0 and cam.max() <= 1.0 and np.isfinite(cam).all()
    out = save_overlay(Image.new("RGB", (100, 80), (10, 200, 30)), cam, tmp_path / "o" / "cam.png")
    assert out.exists() and Image.open(out).size == (64, 64)
    assert all(p.grad is None for p in model.parameters())                # gradients cleaned up after use


def test_image_tool_scores_and_saves_heatmap(tmp_path, monkeypatch):
    model = tiny_classifier()
    monkeypatch.setattr(image_dl, "_load", lambda: {"model": model, "device": torch.device("cpu"),
                                                    "index": {"fake": 0, "real": 1}, "transform": image_dl.eval_transform()})
    img = tmp_path / "input.png"
    Image.fromarray((np.random.rand(90, 120, 3) * 255).astype("uint8")).save(img)
    v = image_dl.score_image(str(img))
    assert not v.error and 0.0 <= v.score <= 1.0 and "raw_logit" in v.raw_features
    assert (tmp_path / "gradcam.png").exists() and v.raw_features["gradcam_path"].endswith("gradcam.png")
    assert not image_dl.score_image(str(img), with_gradcam=False).raw_features.get("gradcam_path")
    bad = image_dl.score_image(str(tmp_path / "missing.png"))
    assert bad.error


def test_image_tool_reports_missing_weights_and_reads_meta(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "image_model_path", str(tmp_path / "nope.pt"))
    image_dl._loaded.clear()
    v = image_dl.score_image(str(tmp_path / "x.png"))
    assert v.error and "Train the model first" in v.explanation
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"")
    weights.with_suffix(".meta.json").write_text(json.dumps({"class_to_idx": {"fake": 1, "real": 0}}))
    assert image_dl._class_index(weights) == {"fake": 1, "real": 0}
    weights.with_suffix(".meta.json").write_text(json.dumps({"class_to_idx": {"cat": 0, "dog": 1}}))
    with pytest.raises(ValueError):
        image_dl._class_index(weights)


def test_binoculars_math_runs_and_is_finite(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from agents.text_agent.tools import slm_binoculars

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=120, n_embd=16, n_layer=1, n_head=2, n_positions=128)
    observer, performer = transformers.GPT2LMHeadModel(cfg).eval(), transformers.GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 120, (1, 60))

    def fake_tokenizer(text, return_tensors=None, truncation=None, max_length=None):
        return transformers.BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    monkeypatch.setattr(slm_binoculars, "_load", lambda: {"tokenizer": fake_tokenizer, "observer": observer,
                                                          "performer": performer, "device": torch.device("cpu")})
    score, n = slm_binoculars.binoculars_score("ignored")
    assert n == 59 and np.isfinite(score) and score > 0
    v = slm_binoculars.score_text("word " * 60)
    assert not v.error and 0.0 <= v.score <= 1.0 and v.raw_features["calibrated"] is False
    assert v.confidence <= 0.5                                            # uncalibrated -> confidence halved
    assert slm_binoculars.score_text("too short").error


def test_binoculars_identical_models_give_ratio_of_perplexity_to_entropy(monkeypatch):
    """With observer == performer, cross-perplexity equals the model's mean entropy; check against a direct computation."""
    transformers = pytest.importorskip("transformers")
    from agents.text_agent.tools import slm_binoculars

    torch.manual_seed(1)
    cfg = transformers.GPT2Config(vocab_size=90, n_embd=16, n_layer=1, n_head=2, n_positions=128)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 90, (1, 50))
    enc = transformers.BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    monkeypatch.setattr(slm_binoculars, "_load", lambda: {"tokenizer": lambda *a, **k: enc, "observer": model,
                                                          "performer": model, "device": torch.device("cpu")})
    score, _ = slm_binoculars.binoculars_score("x")
    with torch.no_grad():
        logp = torch.log_softmax(model(**enc).logits[:, :-1], -1)
        nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1).mean()
        entropy = -(logp.exp() * logp).sum(-1).mean()
    assert score == pytest.approx(float(nll / entropy), rel=1e-4)
