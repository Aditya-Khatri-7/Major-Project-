# How the Forensics Agent works (plain-language guide)

## 1. What it does
You give it a text or an image. It answers one of three things: **authentic**, **synthetic** (AI-made / deepfake), or **uncertain** (a human should look). It also explains why.

## 2. The pipeline in one picture
```
input -> route (text or image)
      -> specialist tools score it (each gives a number 0..1 = "how fake")
      -> RAG: look up similar past cases and reference notes
      -> verifier: combine the scores (fusion) with calibrated thresholds
      -> if the tools disagree or the score is in the grey zone -> "uncertain" + escalate to a human
      -> report (verdict, confidence, evidence) -> saved to the evidence store
```

### Text tools
- **DeBERTa classifier** (`text_dl`): fine-tuned to tell human from AI writing. The main signal.
- **Binoculars** (`text_slm`): compares how surprising the text is to two small language models. Tested: weak, so its weight is capped at 0.10.
- **LLM judge** (`text_llm`): asks a large model for an opinion (needs an API key).

### Image tools
- **EfficientNet-B4** (`image_dl`): trained on real vs fake faces.
- **CLIP probe** (`image_clip`): a frozen CLIP model plus a small linear classifier. Good at generalising to new fakes.
- **Scope guard**: checks the image contains a face; the face tools only apply to faces.
- **Face crop**: the face tools were trained on tight face crops, so we detect the face (MTCNN) and crop it first.

### Fusion
Each tool's score is weighted (weights learned on validation data), averaged, and compared with two thresholds (`t_lo`, `t_hi`) stored in `models/calibration.json`. Between them = uncertain. If tools strongly disagree, the case is escalated.

## 3. Why we test by trying to break it
A single accuracy number on a clean test set hides failures. We deliberately attack the system and count three things:
- **False alarm**: a real item called synthetic.
- **Missed fake**: a synthetic item called authentic.
- **Silent wrong**: wrong verdict that was *not* escalated. This is the dangerous one.

Scripts:
| Script | What it checks |
|---|---|
| `eval/stress_text.py` | short text, typos, caps, repeated text, formal human writing |
| `eval/stress_image.py`, `eval/stress_system_image.py` | padding, rotation, noise, tiny images, watermark, FGSM adversarial attack |
| `eval/system_image_check.py` | full pipeline on AI-edited faces and deepfake sets |
| `eval/smoke_api.py` | 40 live API checks (bad inputs, auth, limits) |

## 4. What went wrong and what we changed
| Failure found | Cause | Fix |
|---|---|---|
| AI-edited ("80s trend") portrait called authentic | training data had face swaps but almost no *edited real photos* | trained on DeepFakeFace (inpainting, swaps, text-to-image), new CLIP probe, retrained CNN, recalibrated |
| Real faces in normal photos flagged | models only saw tight crops | automatic face crop |
| Short text flagged as AI | classifier near chance under ~80 words | under 60 words always "uncertain" |
| Text pasted twice breaks it | repetition confuses the classifier | collapse duplicated blocks |
| Formal human text (speeches, laws, code) called AI | training had little of it | building hard human + matching AI text and retraining |
| Small invisible edits fool the CNN (FGSM) | no adversarial training | adversarial fine-tuning (in progress) + escalate when tools disagree |

## 5. Key ideas to remember
- **Calibration**: a raw score of 0.8 does not mean 80%. We measure on held-out data and set thresholds so the stated confidence is honest.
- **Held-out by identity**: the same face must never appear in both train and test, or the result is inflated (we split by file-name hash).
- **Escalation is a feature**: "uncertain" is better than a confident wrong answer.
- **Limits that remain**: modern generators not in the data, adversarial attacks, very small images, judges needing API quota.

## 6. How to run it
```bash
python -m uvicorn api.main:app --port 8765
python eval/smoke_api.py --base http://127.0.0.1:8765
python -m pytest -q
```
More detail: `WORKLOG.md`, `STRESS_TEST_FINDINGS.md`, `docs/MODELS.md`.
