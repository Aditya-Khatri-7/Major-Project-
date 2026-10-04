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
- The Claude-based judges (LLM, Vision) are implemented but **not benchmarked**: no API key exists in this environment.

## 2. System architecture

```
Streamlit (upload, clipboard paste) -> FastAPI -> LangGraph
  text : DeBERTa-v3-base classifier | Qwen Binoculars | (Claude judge, needs key)
  image: scope guard (is it a face photo?)
           face photo  -> EfficientNet-B4 (+Grad-CAM) | unified CLIP face probe
           other image -> general AI-image probe (CLIP)
           (Claude Vision judge, needs key)
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

### 5.1 Text

| Test set | What it tests | v1 AUC | **Final AUC** | Final acc |
|---|---|---|---|---|
| HC3 test | in-domain | 1.000 | 1.000 | 0.989 |
| MAGE test | many generators and domains | 0.658 | **0.971** | 0.904 |
| RAID test documents | all generators and attacks | 0.721 | **0.987** | 0.952 |
| RAID unseen generator (cohere) | generator never trained on | 0.809 | **0.958** | 0.903 |
| RAID unseen attacks | paraphrase / homoglyph / synonym | 0.655 | **0.974** | 0.925 |

v1 = DeBERTa-small trained on HC3 only. The unseen-attack jump (0.878 -> 0.974, accuracy 0.73 -> 0.925) comes from input
normalisation, with no retraining.

### 5.2 Images: face-swap deepfakes

| Test set | AUC |
|---|---|
| WildDeepfake test | 0.974 |
| FF++ (seen methods, unseen videos) | 0.978 |
| Celeb-DF (official Test split) | 0.998 |
| FF++ unseen methods (Face2Face, FaceShifter) | 0.755 (about 0.79 with CLIP fusion) |

Single-source baseline (WildDeepfake only): FF++ 0.52, Celeb-DF 0.69. Root causes fixed: FF++ frames were full video
frames rather than face crops (preprocessing mismatch), and training used a single source.

Note: these benchmark numbers were measured before the CLIP probe was replaced by the unified one (section 5.3). The unified
probe alone is 0.02-0.03 AUC lower on face-swap sets; a later 480-image offline check of the fused pair gave swap AUC 0.921.
The full benchmark was not rerun after the swap.

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

### 5.4 Full pipeline (local tools, no API key)
Text calibration: DL AUROC 0.995, Binoculars 0.756, fused 0.989 (the weak zero-shot Binoculars slightly lowers the fused
score; reported as is). Smoke test through LangGraph: 6/6 text and 6/6 image cases correct, every report with RAG citations.
Demo check: 17/20 texts from an unseen generator correct (2 of the 3 misses flagged "uncertain" and escalated).

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
| Negative results (kept): self-blended images, extra manipulation types, CLIP-only for unseen methods | method-specific fingerprints | documented in `WORKLOG.md` |

## 7. Testing
66 unit and integration tests pass (`pytest tests -q`), including routing tests for the scope guard. No tests need models or
an API key. `ruff` is clean on files added in this phase (two harmless duplicate-key lint notes remain in
`agents/text_agent/normalize.py`).

## 8. How to run

```powershell
python -m uvicorn api.main:app --port 8000
python -m streamlit run frontend/app.py          # http://localhost:8501, upload or paste an image from the clipboard
```
Demo inputs: `demo_samples/`. Reproduction: `TRAINING_GUIDE.md`; text baseline `python eval/baseline_local.py <tag> text`.
Results tables: `eval/results/*.md`. Presentation script: `RESULTS_AND_PRESENTATION.md`.

## 9. Limitations (honest)

- **Unseen deepfake methods:** about 0.75-0.79 AUC (Face2Face, FaceShifter). Self-blended images, extra manipulation types
  and CLIP were tried; only CLIP fusion helped, modestly.
- **AI-synthesised faces:** improved but not solved: 63% caught at 7% false alarms offline; 12/25 on the harder GRAVEX set.
- **AI art:** AUC 0.999 is in-domain for that dataset; unseen generators will score lower. The general probe is not
  calibrated and is weak on face swaps (AUC 0.59-0.65), so faces are not sent to it.
- **Celeb-DF and FF++ were in the training mix** (video-disjoint tests), so those scores are per-dataset, not zero-shot. The
  only true zero-shot image test is Face2Face / FaceShifter.
- **FF++ data is the C32 compression**, not C23 as the original guide assumed.
- **Claude judges are not benchmarked:** no API key exists here, so the LLM and Vision tools, Reflexion and the full
  ablation table were not run. With fewer than two usable tools, images are always escalated to a human.
- **Full image benchmark not rerun** after the unified CLIP probe replaced the old one.
- Detection is probabilistic, not proof; compression and adversarial edits reduce accuracy.

## 10. Next steps
1. Add an `ANTHROPIC_API_KEY` to `.env`, then run calibration, Reflexion and the full benchmark with a small `--limit` first.
2. Rerun the face-swap image benchmark with the unified probe and refit image calibration thresholds.
3. Add a second, unseen synthetic-face generator set for a true zero-shot synthetic-face test.
4. Retrain the face-swap CNN with synthetic faces included, rather than relying on fusion alone.
