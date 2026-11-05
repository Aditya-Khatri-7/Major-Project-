# Forensics Agent: progress report (5 Nov 2026)

## 1. Summary
The system was tested by trying to break it. Three big weaknesses were found and fixed with new data and retraining; one robustness experiment was run and rejected; several weaknesses remain and are listed honestly in section 6. All results are on held-out data. Everything is pushed to GitHub (commits `c6326cf`, `938f554`, `0a2b9e5`). Unit tests: 83/83. Live API smoke and fail test: 40/40.

## 2. The failure that started it
An AI-edited ("80s trend") portrait was judged authentic. Reproduced: about 60% of AI-edited faces were called authentic. Cause: training contained face swaps and fully fake faces but almost no *edited real photos*.

## 3. What was changed
| Area | Change |
|---|---|
| Image data | DeepFakeFace (real faces, inpainting edits, face swaps, text-to-image), split by identity so no face is in both train and test |
| Image models | New CLIP probe (v3), EfficientNet-B4 retrained with rotation and noise augmentation (v4), recalibrated thresholds |
| Image pipeline | Automatic face detection and crop (also finds sideways faces) before scoring |
| Text pipeline | Under 60 words always "uncertain"; pasted-twice text collapsed |
| Text data and model | 6,000 hard human chunks (bills, government reports, classic books, Python code) and about 4,600 matching AI texts from local Qwen 1.5B/3B; DeBERTa fine-tuned (v3); thresholds recalibrated |
| Testing | `eval/stress_text.py`, `stress_image.py`, `stress_system_image.py`, `system_image_check.py`, `hard_text_eval.py`, `smoke_api.py` |

## 4. Results (before to after)
| Test | Before | After |
|---|---|---|
| AI-edited faces called authentic | about 60% | 0-3% (AUC 0.97-0.99) |
| Accuracy when the system decides (AI-edited faces) | not measured | 96-98% |
| Real face inside a large photo, false alarms | 24% | 0% |
| Face rotated 90 degrees, false alarms | 64% | 28% |
| 25-word human text falsely accused | 68% | 0% (escalated) |
| Text pasted twice, AUC | 0.83 | 0.99 |
| Hard formal human text called AI (own test set) | 64% | 1.3% |
| Ordinary human text falsely accused | 7% | 3% |
| Celeb-DF AUC | 0.99 | 0.999 |

## 5. Experiments that did not ship
- **Adversarial image model (v5):** under a one-step FGSM attack, AUC rose from 0.36 to 0.92 and silent wrong calls fell from 11% to 3%, but FF++ unseen AUC fell from 0.78 to 0.70. Not installed. Weights archived.
- **First text retrain without recalibration:** false alarms on ordinary human text rose (7% to 12%). Caught by the stress test, fixed by recalibration.

## 6. Known weaknesses (honest list)
1. Degraded AI text slips through more than before: lowercase 31% missed (was 21%), typos 14% (was 9%).
2. Legal boilerplate and recipe lists (human) are still called AI.
3. FF++ unseen video deepfakes: AUC 0.78 only.
4. A one-step adversarial attack still breaks the installed image model (11-15% wrong without escalation). Multi-step attacks untested.
5. 15-degree tilt gives about 24% false alarms; tiny images about 14%.
6. Not tested: the user's real 80s-trend image, ChatGPT/Gemini-style edits, newer generators, the LLM judges (Gemini quota).
7. The hard-text test set shares AI generators with training, so its numbers are optimistic.

## 7. Suggested next steps
1. Run the real 80s-trend image and similar edits through the system.
2. Add human recipe, contract and forum text; add lowercase/typo augmentation to text training.
3. Mix FF++ frames into adversarial training and test with multi-step (PGD) attacks.
4. Update the mid-semester report and slides with sections 4-6 (Word/PowerPoint files were not touched).

Details: `STRESS_TEST_FINDINGS.md`, `WORKLOG.md`, `docs/HOW_IT_WORKS.md`.
