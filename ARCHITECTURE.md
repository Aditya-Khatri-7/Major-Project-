# Forensics Agent: Architecture (as built, v2)

Agentic AI + RAG system that decides whether a **text** is AI-generated or an **image** is a deepfake / AI-generated,
explains why, cites retrieved evidence, and escalates uncertain cases to a human.

This file describes the system **as it exists in the code**. The original week-1 plan is kept for reference in
[docs/ARCHITECTURE_v1_original_plan.md](docs/ARCHITECTURE_v1_original_plan.md); every deviation from it is recorded with its
reason in [DECISIONS.md](DECISIONS.md). Measured results: [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

---

## 1. Pipeline

```
Client (Streamlit)
   |  POST /analyze/text {text}        POST /analyze/image (multipart upload)
   v
FastAPI   key check -> rate limit -> validate (type by content, size, dimensions, length) -> job id -> queue slot (503 if busy)
   v
LangGraph StateGraph
   START -- route(modality) --+--> text_agent  --+
                              +--> image_agent --+
                                                 v
                                            rag (retrieve + cite)
                                                 v
                                            verifier --(tools disagree, retries == 0, a judge works)--> reflexion --+
                                                 ^------------------------------------------------------------------+
                                                 v
                                             report  ->  persist (SQLite audit record + case memory)  ->  END
```

One node per stage. State is a typed dict (`agents/schemas.py::ForensicState`); every key a node writes must be declared there.

## 2. Tools

### Text agent (`agents/text_agent/`)

| Tool | Model | Notes |
|---|---|---|
| `text_dl` | DeBERTa-v3-**base**, fine-tuned on a 136k-row mix of HC3 + MAGE + RAID (`models/text_dl_v2`) | sliding windows for long text, temperature-scaled; input first passes `normalize.py` |
| `text_slm` | Binoculars score from Qwen2.5-0.5B (observer) and -Instruct (performer) | needs >= 50 words; logistic map fitted on the calibration split; **fusion weight capped at 0.10** |
| `text_llm` | LLM judge (Anthropic or Gemini, temperature 0) | document wrapped in `<document>` tags, treated as data; robust JSON parsing, never raises |

`normalize.py` strips zero-width characters, applies NFKC, and maps look-alike (Cyrillic/Greek) letters to ASCII when the text is
mostly Latin. The amount of change is reported as tamper evidence. It lifted the homoglyph attack from AUC 0.72 to 0.99.

### Image agent (`agents/image_agent/`)

A **scope guard** (`scope.py`, zero-shot CLIP prompts) first decides whether the image is a face photo, because the deepfake detectors only
know faces.

| Image type | Tools used |
|---|---|
| Face photo | `image_dl` (EfficientNet-B4 on WildDeepfake + FF++ + Celeb-DF, with Grad-CAM) and `image_clip` (frozen CLIP ViT-L/14 + logistic probe, trained on face swaps + AI-synthesised faces), plus `image_vlm` |
| Anything else (art, scenes, logos) | `image_general` (CLIP probe trained on real-vs-AI images), plus `image_vlm`; the face tools return an explanatory "not used" verdict |

`image_vlm` is a vision-language judge (Anthropic or Gemini). Judges are optional: with no key the pipeline runs on local tools only.

## 3. Fusion and verifier (`agents/fusion.py`, `agents/verifier_agent.py`)

```
usable   = tools with error == False and confidence >= 0.15
weight_i = w[tool] * length_factor * confidence_i          (w from validation AUROC, frozen in models/calibration.json)
fused    = sum(weight_i * score_i) / sum(weight_i)
spread   = max(score) - min(score) over usable tools
```

* **Verdict:** `fused >= t_hi` synthetic, `fused <= t_lo` authentic, otherwise uncertain. Thresholds come from the calibration split, never the test sets.
* **Confidence:** `|fused - 0.5| * 2`, minus 0.15 when `spread > 0.35`, capped at 0.95.
* **Reflexion (max 1):** if `spread > 0.35`, a judge is working and a key exists, the judge is re-run once with the peers' scores and retrieved evidence.
* **Escalate to a human** when fewer than two tools are usable, the verdict is uncertain, the tools still disagree, or a *human-verified* case neighbour (similarity >= 0.85) contradicts the verdict.
  Exception: for non-face images the general probe is the designated sole detector, so a decisive verdict (confidence >= 0.5, capped at 0.8) is not force-escalated.
* **RAG never changes a score.** It grounds the report and can only add an escalation flag.

## 4. RAG (`rag/`)

ChromaDB persistent client, cosine space, embedding model `all-MiniLM-L6-v2` on CPU, telemetry off.

* Layer 1: literature notes. Layer 2: generator fingerprints (both ingested from `rag/knowledge_base/`; the API ingests them automatically on first start).
* Layer 3: case memory. Every job is stored `verified=false` (verdict and tool explanations only, never the analysed content); a human correction flips it to `verified=true`.
* CRAG-lite: hits under similarity 0.5 are dropped; if none remain the query is rewritten once; if still none the report says no evidence was retrieved.
* Every report lists the `citations` it used.

## 5. API, storage, security

| Endpoint | Purpose |
|---|---|
| `POST /analyze/text` | body `{text}`, 20 to 20,000 words |
| `POST /analyze/image` | multipart upload; JPEG/PNG/WebP sniffed by content, <= 10 MB, <= 4096 px |
| `GET /jobs/{id}`, `GET /jobs/{id}/gradcam` | stored report, Grad-CAM image |
| `POST /jobs/{id}/correction` | human label; updates case memory |
| `GET /health` | liveness and which models/keys are present |

* Hardening (`api/security.py`, all in `config.Settings`): optional `API_KEY` header, per-client rate limit, 503 when the single GPU slot is busy, optional retention purge of uploaded files (`JOB_RETENTION_DAYS`).
* Evidence store (SQLite via SQLAlchemy, Postgres by URL): job id, SHA-256 of the input, tool verdicts, fused score, verdict, citations, model versions + calibration hash, any human correction. Raw content is not stored.
* Structured JSON log line per node (`observability.py`).
* Config: one `Settings` class (`config.py`), `.env` loaded for every module; judge provider chosen by `LLM_PROVIDER` (`anthropic` | `gemini`).

## 6. Calibration and evaluation

* `training/calibrate.py` fits temperatures, weights and thresholds on a dedicated calibration split and writes `models/calibration.json` (the only place they are set; the file is committed).
* `eval/score_dataset.py` runs each tool once per sample and caches the verdicts; `eval/run_benchmark.py` computes every ablation row offline with bootstrap CIs, TPR@1%FPR, ECE and accuracy-vs-coverage.
* Leakage control: text split by source document/question, images by video, held-out generators and attacks never trained on, automatic overlap assertions in `data/prepare_text_mix.py`.

## 7. Repository layout

```
agents/      orchestrator, verifier, fusion, calibration, reflexion, report, llm_utils, gemini_client, text_agent/, image_agent/
api/         main (endpoints), security (key, rate limit, retention), validation, schemas
rag/         ingest, retriever, store, case_memory, knowledge_base/
evidence_store/  SQLAlchemy audit records, hashing
training/    train_text_classifier, train_image_model, calibrate
eval/        score_dataset, run_benchmark, make_tables, clip probes, plots, results/
data/        dataset preparation scripts (datasets themselves are not in git)
frontend/    Streamlit app
tests/       unit + integration tests (no models or API keys needed)
```

## 8. Hardware

Developed on an RTX 5080 16 GB (the original plan assumed a 3050 6 GB). Peak inference VRAM is a few GB; training two EfficientNet-B4 runs at once
exhausts 16 GB, so jobs run one at a time (queue slot).

## 9. Known limits (see report section 9)

Unseen deepfake methods (AUC about 0.75-0.79), AI-synthesised faces only partly solved, and the judge tools / Reflexion are implemented and unit-tested
but not benchmarked until a judge API key is configured. Detection is probabilistic, not proof.
