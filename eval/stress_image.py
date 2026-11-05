"""Image stress test: perturbations the earlier robustness test did not cover, plus a white-box adversarial attack.

    python eval/stress_image.py --data data/processed/image/tests/celebdf --name celebdf --n 100

Per perturbation it reports fused AUROC, fake recall and real false-alarm rate at the frozen threshold, and how often the scope guard
wrongly decides 'not a face' (which silently switches the face detectors off). The FGSM rows attack image_dl only (white box) and
measure transfer to the CLIP probe and the fused score.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.fusion import fuse  # noqa: E402
from eval.datasets import load_image_records  # noqa: E402

TMP = Path("eval/cache/_stress_tmp")
RNG = np.random.default_rng(0)


def _pad(im: Image.Image, factor: float = 1.8) -> Image.Image:
    w, h = im.size
    canvas = Image.new("RGB", (int(w * factor), int(h * factor)), (96, 110, 120))
    canvas.paste(im, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


def _center_crop(im: Image.Image, frac: float = 0.65) -> Image.Image:
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    return im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))


def _noise(im: Image.Image, sigma: float = 8.0) -> Image.Image:
    a = np.asarray(im).astype(np.float32) + RNG.normal(0, sigma, (im.height, im.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _tiny(im: Image.Image, px: int = 64) -> Image.Image:
    return im.resize((px, px), Image.BILINEAR).resize(im.size, Image.BICUBIC)


def _watermark(im: Image.Image) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    for i in range(3):
        d.text((8, im.height * (i + 1) // 4), "SAMPLE  @news_channel  SAMPLE", fill=(255, 255, 255))
    return im


PERTURB = {
    "original": lambda im: im,
    "pad_1.8x_context": _pad,
    "center_crop_65pct": _center_crop,
    "rotate_90": lambda im: im.rotate(90, expand=True),
    "rotate_15": lambda im: im.rotate(15, expand=False, fillcolor=(0, 0, 0)),
    "grayscale": lambda im: im.convert("L").convert("RGB"),
    "gauss_noise_s8": _noise,
    "tiny_64px": _tiny,
    "watermark_text": _watermark,
    "dark_0.4": lambda im: ImageEnhance.Brightness(im).enhance(0.4),
    "low_contrast_0.5": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
}


def fgsm(path: str, eps: float, to_real: bool) -> Image.Image:
    """One-step white-box attack on image_dl in 380x380 pixel space; pushes the fake-minus-real logit down (or up)."""
    import torch

    from agents.image_agent.tools import dl_classifier as dl

    ctx = dl._load()
    model, device, idx = ctx["model"], ctx["device"], ctx["index"]
    im = Image.open(path).convert("RGB").resize((dl.IMG_SIZE, dl.IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device).requires_grad_(True)
    mean = torch.tensor(dl.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(dl.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    logits = model((x - mean) / std).float()
    score = logits[0, idx["fake"]] - logits[0, idx["real"]]
    score.backward()
    step = -x.grad.sign() if to_real else x.grad.sign()
    adv = (x + eps * step).clamp(0, 1).detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((adv * 255).round().astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=Path("eval/results/stress"))
    args = ap.parse_args()

    from agents.image_agent.scope import face_probability
    from agents.image_agent.tools import clip_probe, dl_classifier

    cal = load_calibration()
    t_star = cal["thresholds"]["image"]["t_star"]
    samples = load_image_records(args.data, "all", args.n, 42)
    labels = np.array([s["label"] for s in samples])
    TMP.mkdir(parents=True, exist_ok=True)
    results = {}
    jobs = [(k, f, None) for k, f in PERTURB.items()]
    jobs += [("fgsm_eps2_to_real", None, 2 / 255), ("fgsm_eps4_to_real", None, 4 / 255), ("fgsm_eps4_random_sign", None, -4 / 255)]
    try:
        for name, fn, eps in jobs:
            fused, dl_s, clip_s, face_p = [], [], [], []
            for i, s in enumerate(samples):
                p = TMP / f"{i}.png"
                if eps is None:
                    with Image.open(s["path"]) as im:
                        fn(im.convert("RGB")).save(p)
                elif eps > 0:
                    fgsm(s["path"], eps, to_real=s["label"] == 1).save(p)
                else:                                         # control: same size of noise, random direction
                    with Image.open(s["path"]) as im:
                        a = np.asarray(im.convert("RGB").resize((380, 380))).astype(np.float32)
                    a += (-eps * 255) * RNG.choice([-1, 1], a.shape)
                    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(p)
                v_dl = dl_classifier.score_image(str(p), with_gradcam=False)
                v_clip = clip_probe.score_image(str(p))
                dl_s.append(v_dl.score)
                clip_s.append(v_clip.score)
                face_p.append(face_probability(str(p)))
                f = fuse([v_dl, v_clip], "image", 0, cal).fused
                fused.append(0.5 if f is None else f)
            fa = np.array(fused)
            results[name] = {
                "fused_auroc": float(roc_auc_score(labels, fa)),
                "dl_auroc": float(roc_auc_score(labels, dl_s)),
                "clip_auroc": float(roc_auc_score(labels, clip_s)),
                "fake_recall": float((fa[labels == 1] >= t_star).mean()),
                "real_false_alarm": float((fa[labels == 0] >= t_star).mean()),
                "scope_says_not_face": float((np.array(face_p) < 0.5).mean()),
            }
            r = results[name]
            print(f"{name:22s} AUC {r['fused_auroc']:.3f} (dl {r['dl_auroc']:.3f} clip {r['clip_auroc']:.3f})  "
                  f"fake recall {r['fake_recall']:.2f}  false alarm {r['real_false_alarm']:.2f}  "
                  f"scope-miss {r['scope_says_not_face']:.2f}", flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"n": len(samples), "t_star": t_star, "results": results}
    (args.out_dir / f"image_stress_{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
