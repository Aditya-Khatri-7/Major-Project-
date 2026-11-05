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
