# Stress-test findings (5 Nov 2026)

Goal: break the system on purpose to find what to fix. Scripts: `eval/stress_text.py`, `eval/stress_image.py`. Raw numbers: `eval/results/stress/`.
Judges were off (no API key), so this tests the local tools plus fusion. Text: 120 human + 120 AI held-out MAGE texts (>= 80 words).
Images: 100 per class from three held-out face sets, frozen thresholds. Small samples: treat differences under about 0.03 AUC as noise.

## Text

| Perturbation | AUC | Human texts called synthetic | AI texts called authentic |
|---|---|---|---|
| original | 0.987 | 7% | 3% |
| first 50 words | 0.794 | 48% | 11% |
| first 25 words | 0.550 | 68% | 15% |
| text repeated twice | 0.827 | 31% | 3% |
| emoji + hashtags | 0.975 | 16% | 3% |
| lowercase, no punctuation | 0.979 | 1% | 21% |
| ALL CAPS | 0.983 | 1% | 18% |
| typos 8%, synonym swap, sentence shuffle | 0.980-0.989 | 2-7% | 3-9% |

Human-written edge cases (all should be "authentic"):
- Called **synthetic, not escalated**: Gettysburg Address (0.89), legal boilerplate (0.98), Python snippet (0.98), recipe (0.99), repetitive casual chat (0.87).
- Correct: ESL English, scientific abstract, Roman-script Hindi, Spanish, poem.
- Mixed text: fused score rises smoothly with AI share (0.13 / 0.34 / 0.52 / 0.67 / 0.91 at 0 / 25 / 50 / 75 / 100%), but the output is still a single binary-style verdict.

## Image (fused AUC; false-alarm rate on real faces)

| Perturbation | Celeb-DF | FF++ seen | FF++ unseen |
|---|---|---|---|
| original | 0.999 (4%) | 0.957 (18%) | 0.825 (8%) |
| photo with 1.8x background (not a face crop) | 0.990 (24%) | 0.728 (52%) | 0.700 (50%) |
| rotated 90 degrees | 0.947 (30%) | 0.842 (34%) | 0.753 (22%) |
| Gaussian noise | 0.994 (2%) | 0.857 (34%) | 0.697 (32%) |
| 64 px then upscaled | 0.978 (16%) | 0.756 (50%) | 0.608 (48%) |
| text watermark | 0.967 (10%) | 0.898 (14%) | 0.787 (10%) |
| grayscale, centre crop, dark, low contrast | about original | about original | about original |
| **FGSM attack, eps 2/255 on the CNN** | **0.591** | **0.252** | **0.010** |

The CLIP probe is almost unaffected by the FGSM attack on the CNN (AUC 0.67-0.90), but the fused score still collapses because the CNN carries the most weight. Random noise of the same size does nothing (AUC 0.77-0.996), so the damage comes from the attack direction, not the noise.

## What is wrong, ranked

1. **Short text is unreliable but accepted.** The API allows 20 words; under about 80 words the classifier is near chance and flags most humans. Fix: raise the minimum or return "too short to judge", and show a length warning.
2. **Confident false accusations on formal, templated or repetitive human text** (famous speeches, legal text, code, recipes). These are the worst failures for a forensic tool: wrong, confident, not escalated. Fix: add such human text to training (legal, code, recipes, speeches, forum chat), and treat very high confidence on templated text as a flag for review.
3. **Uncropped face photos are mishandled.** Models were trained on face crops; a normal photo with background gives 24-52% false alarms. Fix: detect and crop the face before scoring (a face detector is already implied by the scope guard).
4. **No defence against adversarial perturbation.** The CNN is fully invertible with a 2/255 step. Your documents say only "adversarial edits reduce accuracy". Fix: say so plainly in the report; add adversarial training or input preprocessing; make the verifier escalate when the CNN and CLIP disagree strongly (not measured here whether it already does).
5. **Binoculars adds nothing.** AUC 0.50-0.65 in every condition, so the "independent second opinion" and the disagreement check for text are effectively DeBERTa alone. Fix: replace it with a stronger independent text signal, or drop it from the claims.
6. **Repetition and casing.** Duplicating a text drops AUC to 0.83; lowercasing or capitalising loses about 20% of AI texts. Fix: normalise repetition and case before scoring, or train with these augmentations.
7. **Noise, rotation and tiny images** push false alarms to 30-50% on FF++. Fix: rotation/noise/low-resolution augmentation, and a low-resolution warning.

## Where the documents overclaim or omit

- "Text AUC 0.96-0.99 on unseen generators" holds only for full-length text; short text is not reported.
- The limitations section lists JPEG/screenshot damage but not uncropped photos, rotation, noise or adversarial attack.
- Binoculars is described as an independent second opinion; the data shows it is not informative.

## Not tested here (needs more work)

- LLM judges and Reflexion (no working key; the earlier judge failure on the demo deepfake stands).
- Newer image generators and non-face AI images (needs the extra datasets discussed earlier).
- Whether the verifier escalates under the FGSM attack, and the full API path (upload limits, malformed files).
- Real paraphrasing by a language model; only crude synonym swaps were used.

## Update after fixes (5 Nov 2026, evening)

Re-ran the same tests on the retrained models (`eval/stress_system_image.py`, `eval/smoke_api.py`, `eval/stress_text.py`).

| Flaw | Before | After |
|---|---|---|
| AI-edited faces (DeepFakeFace test) called authentic | about 60% | 0-3%; fused AUC 0.97-0.99 |
| Short text (25 words), human texts called synthetic | 68% | 0% (always "uncertain", escalated) |
| Text pasted twice, AUC | 0.83 | 0.99 |
| Real face inside a large photo, false alarms (Celeb-DF) | 24% | 0% |
| Face rotated 90 degrees, Celeb-DF false alarms | 64% | 28% (AUC 0.84 -> 0.94) |
| FGSM attack: wrong verdict that is also NOT escalated | not measured | 11-15% (70-76% of attacked cases escalated) |
| Binoculars uninformative | AUC 0.5-0.65 | unchanged (still capped at weight 0.10) |
| Confident false accusation of formal human text | 5 of 10 edge cases | 4 of 10 (code and Spanish now escalated; Gettysburg, legal, recipe, repetitive chat unchanged) |

Still open, in priority order: formal/templated human text (needs hard negatives plus matching AI text and a DeBERTa retrain), white-box adversarial robustness, 15-degree tilt (24% false alarms on Celeb-DF), tiny images, ChatGPT/Gemini-style image edits, judges.

## Round 3: adversarial training of the image CNN (v5), 5 Nov 2026

Fine-tuned v4 for 1 epoch with FGSM adversarial examples mixed into half the batches (eps 3/255). White-box FGSM eps 2/255 on Celeb-DF, full system:

| | v4 (installed) | v5 (adversarial) |
|---|---|---|
| Attacked AUC | 0.36 | 0.92 |
| Attacked, wrong and NOT escalated | 11% | 3% |
| Attacked, escalated | 70% | 96% |
| Celeb-DF clean AUC | 0.999 | 0.998 |
| DFF inpainting / insight / text2img AUC | 0.985 / 0.968 / 0.990 | 0.985 / 0.967 / 0.995 |
| FF++ seen / unseen AUC | 0.925 / 0.776 | 0.898 / 0.698 |
| FF++ unseen fakes called authentic | lower | 42% |

Decision: **v4 stays installed.** v5 buys robustness to a one-step attack but loses accuracy on compressed video-style deepfakes (FF++). v5 was trained against the same attack it is tested with, so the gain may not hold against multi-step attacks (PGD); that was not tested. Weights kept in `models/efficientnet_b4_v5.pt` and calibration in `models/_archive/calibration_v5.json`. A fairer try would mix FF++ frames into the adversarial training set.

## Round 4: hard human text and a text retrain (v3), 5 Nov 2026

Built 6,000 hard human chunks (bills, government reports, classic books, Python code) and about 4,600 matching AI texts from local Qwen 1.5B and 3B. Fine-tuned DeBERTa v2 for 2 epochs on this plus a replay sample of the original mix (`data/build_hard_mix.py`, `models/text_dl_v3`). Evaluated with `eval/hard_text_eval.py`.

| Held-out test | v2 | v3 |
|---|---|---|
| Hard human texts called synthetic (legal / code) | 64% (97% / 96%) | 2.7% (0% / 1%) |
| Hard set AUC | 0.971 | 0.999 |
| MAGE test AUC (regression check) | 0.968 | 0.971 |

Caveat: the hard test set shares AI generators and prompt styles with training, so it overstates the gain. The 10 hand-written edge cases are the independent check:

| Edge case | v2 | v3 |
|---|---|---|
| Gettysburg Address (human) | synthetic (0.88) | authentic (0.12) |
| Python code with comments | 0.99 | uncertain (0.29) |
| Legal boilerplate | synthetic | **still synthetic (0.99)** |
| Recipe list | synthetic | **still synthetic (1.00)** |
| Repetitive casual chat | synthetic | **still synthetic (0.82)** |
| ESL, abstract, Hindi (Roman), poem | correct | correct |

Installed v3 (`config.py`, `.env`). Text thresholds were not recalibrated. Still open: recipes, contract boilerplate and forum chat need their own human training text; Binoculars adds no signal; judges untested.

### Round 4 addendum: v3 recalibrated (text thresholds 0.685/0.785 -> 0.836/0.936)

Installing v3 with the old thresholds raised false alarms on ordinary human text (7% -> 12%; emoji text 16% -> 32%): v3 is more confident, so v2's thresholds were too low. After recalibrating on `mix_calib` (1,500 texts, Binoculars weight capped at 0.10):

| Condition | v2 false alarm / AI missed | v3 recalibrated |
|---|---|---|
| Original MAGE text | 7% / 3% | 3% / 2% |
| Emoji + hashtags | 16% / 3% | 13% / 1% |
| Text pasted twice | 7% / 3% | 3% / 3% |
| Hard formal human text (own test set) | 64% | 1.3% |
| Lowercase, no punctuation: AI missed | 21% | **31%** |
| 8% typos: AI missed | 9% | **14%** |
| ALL CAPS: AI missed | 18% | **22%** |

Trade-off: fewer false accusations everywhere, but more AI text slips through when it is degraded (lowercased, typos). Edge cases: Gettysburg and repetitive chat now authentic; legal boilerplate and recipe still wrongly synthetic. Raw numbers: `eval/results/stress/text_stress_v3.json`.
