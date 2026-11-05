"""Evaluate public Hugging Face AI-image detectors on OpenFake-tiny test and on the user's two images (inference only)."""
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModelForImageClassification

warnings.filterwarnings("ignore")
IDS = sys.argv[1:] or ["haywoodsloan/ai-image-detector-deploy", "Organika/sdxl-detector", "Ateeqq/ai-vs-human-image-detector",
                       "umm-maybe/AI-image-detector", "Smogy/SMOGY-Ai-images-detector"]
D = "D:/Temp/claude/c4a934cafebe8ad0/2d7fcf68-2a16-4b8d-9ff9-ad452b889af2/"
mine = {"bedroom_FAKE": D + "images/1.jpg", "street_REAL": D + "scratchpad/img2.png"}
meta = pd.read_csv("data/processed/image/openfake_tiny/meta.csv")
te = meta[meta.split == "test"].groupby("label", group_keys=False).apply(lambda g: g.sample(250, random_state=0))
for mid in IDS:
    try:
        proc = AutoImageProcessor.from_pretrained(mid)
        model = AutoModelForImageClassification.from_pretrained(mid, use_safetensors=True).cuda().eval().half()
    except Exception as e:
        print(mid, "LOAD FAIL", str(e)[:100]); continue
    labels = {i: name.lower() for i, name in model.config.id2label.items()}
    fake_idx = [i for i, n in labels.items() if any(k in n for k in ("ai", "fake", "artificial", "gen", "synthetic"))
                and "human" not in n and "real" not in n]
    if not fake_idx:
        print(mid, "cannot find fake label in", labels); continue

    def score(path):
        x = proc(images=Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad():
            p = model(pixel_values=x).logits.float().softmax(-1)[0]
        return float(p[fake_idx].sum())
    s = np.array([score(p) for p in te.path])
    y = (te.label == "fake").values
    print(f"{mid:45s} labels={labels} AUC={roc_auc_score(y, s):.3f} | " + " ".join(f"{k}={score(v):.2f}" for k, v in mine.items()), flush=True)
