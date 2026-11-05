"""Create real LOCAL AI edits with Stable Diffusion inpainting, with exact ground-truth masks.

    python eval/make_local_edits.py --n 400 --split test --out data/processed/image/local_edits
Each real face photo gets one edit of a random size (about 2%..40% of the image), composited back so ONLY the masked region changes.
Output: <out>/<split>/{real,edited,mask}/<id>.png and <out>/<split>/meta.csv (id, area, prompt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from eval.clip_face_v3 import collect  # noqa: E402

PROMPTS = ["a person smiling warmly, photo", "a person wearing round glasses, photo", "a different hairstyle, photo", "a person with a beard, photo",
           "a gold necklace, photo", "a person with different eyes, photo", "a red scarf, photo", "a tattoo on the skin, photo",
           "a person wearing a hat, photo", "natural skin, photo", "a different background object, photo", "a person with different makeup, photo"]
AREAS = [0.02, 0.04, 0.08, 0.15, 0.25, 0.40]


def make_mask(rng, area: float, size: int = 512) -> Image.Image:
    side = (area ** 0.5) * size
    w, h = side * rng.uniform(0.8, 1.25), side * rng.uniform(0.8, 1.25)
    w, h = min(w, size * 0.95), min(h, size * 0.95)
    # bias toward the central face region where edits are most meaningful
    cx = rng.uniform(w / 2, size - w / 2) * 0.5 + size / 4
    cy = rng.uniform(h / 2, size - h / 2) * 0.5 + size / 4
    cx, cy = min(max(cx, w / 2), size - w / 2), min(max(cy, h / 2), size - h / 2)
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    (d.ellipse if rng.random() < 0.6 else d.rectangle)(box, fill=255)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("data/processed/image/local_edits"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "models/gen/sd15-inpainting", torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rng = np.random.default_rng(a.seed)
    wiki = collect("wiki")[a.split]
    stems = sorted(wiki)
    pick = [stems[i] for i in rng.choice(len(stems), a.n, replace=False)]
    root = a.out / a.split
    for sub in ("real", "edited", "mask"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    meta = []
    for k, stem in enumerate(pick):
        real = Image.open(wiki[stem]).convert("RGB").resize((512, 512), Image.BICUBIC)
        area = AREAS[k % len(AREAS)]
        mask = make_mask(rng, area)
        prompt = PROMPTS[int(rng.integers(len(PROMPTS)))]
        gen = torch.Generator("cuda").manual_seed(int(rng.integers(1 << 30)))
        out = pipe(prompt=prompt, image=real, mask_image=mask, num_inference_steps=25, guidance_scale=7.5,
                   generator=gen).images[0].resize((512, 512))
        soft = mask.filter(ImageFilter.GaussianBlur(3))
        edited = Image.composite(out, real, soft)
        sid = f"{a.split}_{k:04d}"
        real.save(root / "real" / f"{sid}.png")
        edited.save(root / "edited" / f"{sid}.png")
        mask.save(root / "mask" / f"{sid}.png")
        meta.append({"id": sid, "area": area, "actual_area": float((np.asarray(mask) > 127).mean()), "prompt": prompt})
        if k % 25 == 0:
            print(k, "/", a.n, flush=True)
            pd.DataFrame(meta).to_csv(root / "meta.csv", index=False)
    pd.DataFrame(meta).to_csv(root / "meta.csv", index=False)


if __name__ == "__main__":
    main()
