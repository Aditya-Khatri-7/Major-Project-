# Forensics Agent: Final Project Report

Agentic AI + RAG system that decides whether a **text** is AI-generated or an **image** is a deepfake / AI-generated,
explains its reasoning, cites evidence, and escalates uncertain cases to a human.

## 1. Executive summary

- **Text detection is strong and generalises:** AUC 0.96-0.99 on generators, domains and attacks never seen in training.
- **Image detection is strong on face-swap deepfakes** seen per dataset (AUC 0.98-0.998) and **now covers non-face images
  (AI art) and AI-synthesised faces**, which the first version missed completely.
- The main research finding: **in-domain scores (0.97-1.0) hide a large generalisation gap.** Our first models dropped to
  AUC 0.66 (text) and 0.52 (images, chance level) on unseen data. Diagnosing and fixing that is the core of this work.
- Remaining weaknesses are documented in section 9, not hidden.
- The LLM and vision judges (Gemini or Claude, switchable) are implemented and unit-tested but **not benchmarked**: no judge API key has been configured yet.

## 2. System architecture

```
Streamlit (upload, clipboard paste) -> FastAPI -> LangGraph
  text : DeBERTa-v3-base classifier | Qwen Binoculars | (LLM judge, needs a Gemini/Claude key)
  image: scope guard (is it a face photo?)
           face photo  -> EfficientNet-B4 (+Grad-CAM) | unified CLIP face probe
           other image -> general AI-image probe (CLIP)
           (vision-LLM judge, needs a key)
  -> RAG (ChromaDB: literature + generator fingerprints + case memory, cited)
  -> verifier (calibrated weighted fusion, disagreement check, uncertain band, one Reflexion retry, escalate flag)
  -> report -> evidence store (SQLite, SHA-256 input hash)
```

Design records: `ARCHITECTURE.md`, `DECISIONS.md`, `WORKLOG.md` (step-by-step history of the image work).

## 3. Data

| Modality | Dataset | Use |
|---|---|---|
| Text | HC3, MAGE, RAID | training mix (136k rows), validation, calibration, held-out tests |
| Image (face swap) | WildDeepfake subset, FaceForensics++ (C32 frames), Celeb-DF v2 | training mix, validation, held-out tests |
| Image (synthetic faces) | GRAVEX-200K, DeepDetect-2025 | unified CLIP face probe |
| Image (art / general) | DeepDetect-2025, real-vs-AI art set | general AI-image probe |

Leakage control: text split **by source document** (RAID) and by question (HC3); images split **by video**; held-out
generators (cohere) and attacks (paraphrase, homoglyph, synonym) never appear in training; automatic leakage assertions in
`data/prepare_text_mix.py`. Old RAID test files that overlapped the training set were retired.

## 4. Models

| Tool | Model | Training |
|---|---|---|
| Text classifier | DeBERTa-v3-base | HC3 + MAGE + RAID mix, early stopping on val AUC (best epoch 1, val AUC 0.989) |
| Text input normalisation | zero-width strip, NFKC, look-alike character mapping | no training; fixes homoglyph attacks |
| Face-swap CNN | EfficientNet-B4 | 3-dataset mix, strong augmentation, early plateau at val AUC 0.982 |
| Unified CLIP face probe | CLIP ViT-L/14 (frozen) + logistic regression | face swaps + GRAVEX + DeepDetect; temperature-calibrated |
| General AI-image probe | CLIP ViT-L/14 (frozen) + logistic regression | DeepDetect + AI-art set; used for non-face images |
| Scope guard | zero-shot CLIP prompts | no training; threshold 0.5 |

## 5. Results (all on held-out data)

### 5.1 Text (final system, current code and calibration)

Each set: 1,000 held-out samples, 95% bootstrap CIs, thresholds frozen from the calibration split. "Full" = classifier + Binoculars fused
by the verifier (Binoculars weight capped at 0.10). Escalation = share of samples sent to a human; accuracy is measured on the rest.

| Test set | What it tests | DeBERTa alone AUC | Binoculars alone AUC | **Full AUC [95% CI]** | TPR@1%FPR | ECE | Escalated | Acc. on decided |
|---|---|---|---|---|---|---|---|---|
| HC3 test | in-domain | 1.000 | 0.961 | **0.998** [0.994-1.000] | 1.00 | 0.014 | 33% | 0.996 |
| MAGE test | many generators, domains | 0.967 | 0.628 | **0.961** [0.947-0.971] | 0.58 | 0.065 | 52% | 0.958 |
| RAID test documents | all generators and attacks | 0.986 | 0.718 | **0.985** [0.978-0.990] | 0.81 | 0.042 | 30% | 0.966 |
| RAID unseen generator (cohere) | generator never trained on | 0.957 | 0.923 | **0.957** [0.946-0.969] | 0.60 | 0.043 | 43% | 0.933 |
| RAID unseen attacks | paraphrase / homoglyph / synonym | 0.968 | 0.621 | **0.962** [0.949-0.972] | 0.61 | 0.068 | 36% | 0.920 |

Reading the table honestly:
* The classifier carries the system. Fusing in Binoculars changes AUC by -0.001 to -0.006 (it never improves AUC on these sets), which is why its
  weight is capped at 0.10: it is kept as an independent second opinion, not as a score booster.
* At a strict 1% false-positive budget the detector catches 58-61% of AI text on the hardest sets (MAGE, unseen attacks) and 81-100% on easier ones.
  A forensic user should read the TPR@1%FPR column, not the AUC column.
* Accuracy on the cases the system actually decides is 92-99.6%; the price is that 30-52% of samples are handed to a human.

v1 (DeBERTa-small, HC3 only) scored MAGE 0.658 and unseen-generator 0.809. The earlier full-set runs of the v2 classifier (MAGE 0.971, RAID
0.987, unseen generator 0.958, unseen attacks 0.974) agree with the numbers above within their confidence intervals. The homoglyph attack
moved from AUC 0.72 to 0.99 through input normalisation, with no retraining.

### 5.2 Images: face-swap deepfakes (final system, rerun after the CLIP probe change and recalibration)

1,500 held-out images per set, thresholds frozen from a separate 2,000-image calibration split. "Full" = EfficientNet-B4 + unified CLIP probe.

| Test set | EfficientNet AUC | CLIP probe AUC | **Full AUC [95% CI]** | TPR@1%FPR | ECE | Escalated | Acc. on decided |
|---|---|---|---|---|---|---|---|
| WildDeepfake test | 0.973 | 0.914 | **0.977** [0.968-0.984] | 0.67 | 0.052 | 29% | 0.977 |
| FF++ (seen methods, unseen videos) | 0.980 | 0.809 | **0.976** [0.970-0.982] | 0.70 | 0.054 | 39% | 0.954 |
| Celeb-DF (official Test split) | 0.999 | 0.901 | **0.999** [0.997-1.000] | 0.99 | 0.078 | 29% | 0.995 |
| FF++ unseen methods (Face2Face, FaceShifter) | 0.753 | 0.699 | **0.794** [0.771-0.816] | 0.13 | 0.185 | 53% | 0.759 |

The single-source baseline (WildDeepfake only) scored FF++ 0.52 and Celeb-DF 0.69. The unified CLIP probe is weaker than EfficientNet on face
swaps (it also covers AI-synthesised faces), so on the seen sets fusion is roughly equal to EfficientNet alone; its value shows on synthetic faces (5.3).
On unseen manipulation methods the system is **not reliable**: at 1% false positives it catches 13% of fakes, and it is poorly calibrated (ECE 0.19).

**Robustness to re-uploads** (`eval/robustness_image.py`, 300 images per set, fused AUC / false-alarm rate on real faces):

| Degradation | Celeb-DF | WildDeepfake | FF++ (seen) |
|---|---|---|---|
| original | 0.999 / 3% | 0.977 / 7% | 0.967 / 15% |
| JPEG q=50 | 0.992 / 5% | 0.975 / 9% | 0.921 / 22% |
| downscale to half | 0.997 / 13% | 0.980 / 11% | 0.897 / 32% |
| screenshot-like (downscale + JPEG 60) | 0.991 / 9% | 0.984 / 17% | 0.899 / 49% |

Ranking quality (AUC) survives on Celeb-DF and WildDeepfake, but the false-alarm rate on real faces rises with degradation, and FF++ degrades most.
Social-media-style recompression therefore needs a human check; it should be stated as a limitation.

### 5.3 Images: AI-synthesised faces and AI art (added after live testing)

A user test with an AI-generated illustration returned "authentic": the face-swap tools had never seen non-face or
synthetic images. Two additions fixed this.

| Capability | Result |
|---|---|
| Unified CLIP probe, GRAVEX synthetic faces | AUC 0.683 -> **0.917** |
| Unified CLIP probe, DeepDetect synthetic faces | AUC 0.922 -> **0.981** |
| General probe, DeepDetect test | AUC 0.968 |
| General probe, AI-art test | AUC 0.999 (in-domain for that dataset) |
| Live API: AI art | 39 / 40 correct, all routed to the general probe |

Synthetic faces through the full face-tool fusion (offline, 120 images per class per source, held out):

| Rule | AUC | Fakes caught | Real faces wrongly flagged |
|---|---|---|---|
| Face-swap CNN only (before) | 0.575 | 21% | 7% |
| **Weighted average of CNN + CLIP (kept)** | **0.910** | **63%** | 7% |
| Max of CNN and CLIP (rejected) | 0.943 | 96% | **20%** |

The max rule was rejected because it wrongly accuses 20-30% of real faces; a forensic tool should not do that. Live API check
of the final system: Celeb-DF deepfakes 25/25 and real 25/25; DeepDetect AI faces 18/25; GRAVEX AI faces 12/25 (harder set);
real faces 22/25 on both sets.

### 5.4 Full pipeline (local tools, no judge key)
Smoke test through LangGraph: 6/6 text and 6/6 image cases correct, every report with RAG citations. Demo check: 17/20 texts from an unseen
generator correct (2 of the 3 misses flagged "uncertain" and escalated). The numbers in 5.1 and 5.2 come from the same tools and the same fusion code
(`eval/run_benchmark.py` calls `agents.fusion.fuse`), so benchmark and live behaviour match.

## 6. Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Text collapsed on MAGE (0.66) | single-source training | mixed training, held-out generators and attacks |
| Images at chance on FF++ (0.52) | full frames vs face crops; single source | face crops, 3-dataset mix, augmentation |
| Homoglyph attack broke text model (0.72) | look-alike characters break the tokenizer | input normalisation (0.99) |
| Illustration called "authentic" | no tool covered non-face images | scope guard + general probe |
| AI-synthesised faces missed | tools knew only face swaps | unified CLIP face probe |
| NaN crashes (text and image training) | fp16 defaults and fp16 model loading | bf16 and fp32 weights |
| Live text tool crashed | removed tokenizer method in transformers 5 | manual special tokens |
| CSV read failure | RAID texts above the 131 KB field limit | raised limit |
| Possible leakage | frames / texts from one source in train and test | split by video / document, automatic checks |
| 75% of MAGE samples escalated | the weak Binoculars tool "disagreed" with the classifier, triggering the spread rule although it carries almost no weight | tools below weight 0.3 are advisory: they no longer trigger disagreement or Reflexion (mentioned in the notes instead) | escalation 75% -> 52% on MAGE, 30-43% elsewhere |
| Image calibration stale after the CLIP probe was replaced | thresholds fitted to the retired probe | re-scored the calibration split, refit, reran all image benchmarks | benchmarks above |
| AI-art images always escalated | only one tool is usable by design for non-face images | designated sole detector may return a decisive verdict | no forced human review for confident art verdicts |
| Docker image would crash on image requests | `scikit-learn`/`joblib` missing from runtime requirements; compose pointed at the retired v1 text model; empty RAG store on a fresh volume | requirements fixed, v2 path, knowledge base auto-ingested on start, CI builds the images and imports the app inside | CI job green |
| Uploads kept forever, open endpoints | no retention, no auth, unbounded queue | optional API key, rate limit, 503 on busy queue, opt-in retention | tested in `tests/test_api_security.py` |
| Negative results (kept): self-blended images, extra manipulation types, CLIP-only for unseen methods | method-specific fingerprints | documented in `WORKLOG.md` |

## 7. Testing
79 unit and integration tests pass (`pytest tests -q`): fusion branches (including the advisory-tool rule), verifier and sole-detector rule, API
validation and security (key, rate limit, queue, retention), Gemini adapter (mocked HTTP), RAG, normalisation, routing. None need models or a key.
`ruff check .` is clean. GitHub Actions runs lint + tests and builds both Docker images, importing the app inside the API image; all runs are green on
`main` and on the `Aditya-Khatri` branch. Docker is not installed on the development machine, so the container build is verified in CI rather than locally.

## 8. How to run

```powershell
python -m uvicorn api.main:app --port 8000
python -m streamlit run frontend/app.py          # http://localhost:8501, upload or paste an image from the clipboard
```
Demo inputs: `demo_samples/`. Reproduction: `TRAINING_GUIDE.md`; text baseline `python eval/baseline_local.py <tag> text`.
Results tables: `eval/results/*.md`. Presentation script: `RESULTS_AND_PRESENTATION.md`.

## 9. Limitations (honest)

- **Unseen deepfake methods:** AUC 0.79 (Face2Face, FaceShifter); only 13% of fakes caught at 1% false positives. Self-blended images, extra
  manipulation types and CLIP were tried; only CLIP fusion helped, modestly.
- **Re-uploaded images:** false alarms on real faces grow with downscaling and recompression (up to 17% on WildDeepfake, 49% on FF++ with
  screenshot-like damage). AUC mostly survives, so a threshold tuned for degraded inputs would help; it was not done.
- **AI-synthesised faces:** improved but not solved: 63% caught at 7% false alarms offline; 12/25 on the harder GRAVEX set.
- **AI art:** AUC 0.999 is in-domain for that dataset; unseen generators will score lower. The general probe is not calibrated (hand-set weight
  0.88 from its held-out AUC) and is weak on face swaps, so faces are not sent to it.
- **Text:** at 1% false positives only 58-61% of AI text is caught on MAGE and unseen attacks. Human writing styles absent from training data
  (for example non-native English, which the knowledge base flags as a known bias) were **not measured**: no such dataset was available.
- **Celeb-DF and FF++ were in the training mix** (video-disjoint tests), so those scores are per-dataset, not zero-shot. The only true zero-shot
  image test is Face2Face / FaceShifter. FF++ is the C32 compression, not C23.
- **Judges and Reflexion are not benchmarked.** The Gemini judges were run live on the four demo inputs (all returned valid verdicts): text AI 0.72, human 0.05;
  vision judge 0.05 on both the deepfake and the real image, i.e. **it missed the deepfake with 85% confidence**. The free-tier key allows 20 requests per day
  per model, which is exhausted by the smoke test, so calibration and the judge ablation rows could not be run. Judge weights are therefore set conservatively
  (0.3, not fitted). With a billing-enabled key: score 400 calibration samples per modality, run `training/calibrate.py`, then the benchmarks. Without a judge, face images with both local tools usable are still escalated when the fused score is in the
  uncertain band.
- Detection is probabilistic, not proof; adversarial edits reduce accuracy.

## 10. Next steps
1. Put `GEMINI_API_KEY` in `.env` (`LLM_PROVIDER=gemini`), run `python eval/smoke_judges.py`, then score the calibration split and a small
   benchmark (`--limit 100`) with the judges enabled to fill the "LLM only / VLM only / full + Reflexion" ablation rows.
2. Tune an image threshold on degraded (re-uploaded) calibration images to cut false alarms on social-media inputs.
3. Add a second, unseen synthetic-face generator set for a true zero-shot test, and a non-native-English human text set for the text false-positive audit.
4. Retrain the face-swap CNN with synthetic faces included rather than relying on fusion alone.
