# Model release set

Weights are not in git (size). Everything below lives in `models/` on the development machine; `models/calibration.json` is committed.

## Used at runtime (the release set)

| File | What | Used by |
|---|---|---|
| `text_dl_v2/` | DeBERTa-v3-base, HC3 + MAGE + RAID mix (HF format) | `text_dl` |
| `efficientnet_b4.pt` + `.meta.json` + `class_to_idx.json` | EfficientNet-B4, WildDeepfake + FF++ + Celeb-DF mix | `image_dl`, Grad-CAM |
| `clip_probe_v2.joblib` | CLIP ViT-L/14 + logistic probe: face swaps + AI-synthesised faces | `image_clip` |
| `clip_general.joblib` | CLIP ViT-L/14 + logistic probe: real vs AI images (non-face) | `image_general` |
| `clip_face_unified.joblib` | training artefact of the unified probe (eval scripts) | `eval/clip_face_unified.py` |
| `calibration.json` | temperatures, fusion weights, thresholds | every tool, verifier |

Also downloaded from the Hugging Face hub at first use (cache in `HF_HOME`): CLIP ViT-L/14, Qwen2.5-0.5B (+ Instruct), all-MiniLM-L6-v2.

## Archived in `models/_archive/` (moved, not deleted)

* `image_experiments/`: WD-only v2, SBI, +Face2Face, mix variants (negative or superseded experiments documented in WORKLOG.md)
* `training_checkpoints/`: `*_last.pt` optimizer/resume checkpoints (about 4 GB)
* `old_versions/`: text v1 (DeBERTa-small), original WildDeepfake model, old CLIP probe, previous calibration files

Safe to delete once the project is submitted.
