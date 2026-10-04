> **Archived.** This is the original week-1 plan. The system as built differs (scope guard, CLIP probes, DeBERTa-base v2, Gemini option). Current design: [../ARCHITECTURE.md](../ARCHITECTURE.md).

# Forensics Agent — Final Architecture & Roadmap (LOCKED)

Agentic AI + RAG system for detecting AI-generated **text** and **images**.
Every decision below is final. Change one only if an experiment disproves it, and record why in `DECISIONS.md`.

---

## 1. Locked decisions

| Area | Decision |
|---|---|
| Orchestration | LangGraph `StateGraph`, typed state, one node per stage |
| Text agent | 3 tools: **DL classifier** (DeBERTa-v3-small) · **SLM zero-shot** (Qwen2.5-0.5B Binoculars) · **LLM judge** (Claude API) |
| Image agent | 2 tools: **DL classifier** (EfficientNet-B4 + Grad-CAM) · **VLM judge** (Claude Vision API) |
| Removed | XGBoost/stylometric ML, GPT-2 pseudo-Binoculars, SD-Turbo reconstruction |
| Fusion | Fixed weights from validation AUROC × length factor. No learned stacker. |
| Verifier | Disagreement check among confident tools, uncertain band, one Reflexion retry, escalate flag |
| RAG | ChromaDB (cosine), 3 layers, citations required in every report |
| Storage | SQLite via SQLAlchemy (Postgres by URL change), SHA-256 input hash, JSON logs |
| API | FastAPI, uploads only (no server paths), size/type validation |
| Frontend | Streamlit |
| Hardware target | RTX 3050 6 GB, 16 GB RAM. Nothing local may exceed ~4.5 GB VRAM. |
| Not doing | Pretraining an SLM from scratch, local VLM, distributed tracing, legal chain-of-custody (all future work) |

---

## 2. Pipeline

```
Client (Streamlit)
   │  POST /analyze/text {text}      POST /analyze/image (multipart upload)
   ▼
FastAPI  ── validate (type, size, length) ── create job_id ── save upload to job dir
   ▼
LangGraph
   START ─ route(modality) ─┬─► text_agent  ─┐
                            └─► image_agent ─┤
                                             ▼
                                        rag_retrieve   (query = modality + tool explanations)
                                             ▼
                                          verifier ──(disagreement & retries==0)──► reflexion_rerun ─┐
                                             ▲──────────────────────────────────────────────────────┘
                                             ▼
                                          report   (verdict + citations + per-tool breakdown)
                                             ▼
                                          persist  (SQLite + case memory in Chroma)
                                             ▼
                                            END
```

---

## 3. State schema (`agents/schemas.py`) — single source of truth

```python
class ToolVerdict(BaseModel):
    tool_name: str
    modality: Literal["text", "image"]
    score: float            # calibrated P(AI/fake), 0..1
    confidence: float       # 0..1, how much to trust this tool on THIS input
    explanation: str
    raw_features: dict = {}
    error: bool = False     # True => excluded from fusion and disagreement

class RetrievedEvidence(BaseModel):
    source_id: str
    content_snippet: str
    similarity_score: float          # cosine similarity, 0..1
    evidence_type: Literal["generator_fingerprint", "case_study", "literature"]

class ForensicState(TypedDict):
    job_id: str
    modality: Literal["text", "image"]
    input_ref: str                   # raw text, or path to uploaded file inside job dir
    tool_verdicts: list[ToolVerdict]
    retrieved_evidence: list[RetrievedEvidence]
    retries: int                     # Reflexion counter (max 1)
    fused_score: Optional[float]     # P(AI) after fusion. Benchmarks and plots use THIS.
    verifier_notes: Optional[str]
    final_verdict: Optional[Literal["authentic", "synthetic", "uncertain"]]
    final_confidence: Optional[float]
    escalate_to_human: bool
    citations: list[str]             # source_ids actually used
    report: Optional[dict]
```

Rule: every key any node writes must be declared here (LangGraph drops undeclared keys).

---

## 4. Text agent

| Tool | Model | Input handling | Output |
|---|---|---|---|
| `dl_classifier` | DeBERTa-v3-small, fine-tuned on HC3 (fp16, batch 8, max 256 tokens, sliding windows averaged for long text) | any length ≥ 20 words | P(AI) after temperature scaling; confidence = `abs(p − 0.5)·2` |
| `slm_binoculars` | Observer `Qwen2.5-0.5B`, performer `Qwen2.5-0.5B-Instruct` (~2 GB fp16), real Binoculars score = log-PPL ÷ cross-perplexity | ≥ 50 words, truncated to 512 tokens | score = logistic map of the Binoculars score, fitted on the validation split. Below 50 words: `error=True`. |
| `llm_judge` | Claude (model id from `LLM_MODEL_ID`, default `claude-sonnet-5-5`), temperature 0 | any length ≥ 10 words | score, confidence, one-sentence reason |

Rules:
- **Prompt-injection hardening.** The input text goes inside `<document>` tags, and the system prompt says to treat it as data, never instructions.
- **JSON parsing is robust.** Strip code fences, validate with Pydantic, retry once on parse failure. On failure return `error=True`. Never raise into the graph.
- **Length factor** applied to fusion weights: SLM ×0.5 at 50–100 words, ×1.0 above. DL ×0.7 below 50 words.

## 5. Image agent

| Tool | Model | Output |
|---|---|---|
| `dl_classifier` | EfficientNet-B4 fine-tuned on WildDeepfake, plus **Grad-CAM** heatmap saved to `jobs/<id>/gradcam.png` | P(AI) after temperature scaling; confidence = `abs(p − 0.5)·2` |
| `vlm_judge` | Claude Vision (`LLM_MODEL_ID`), temperature 0, JSON with `score`, `confidence`, `artifacts_found` | same schema, robust parsing as above |

Image retraining recipe (final): ImageNet init · RandomResizedCrop(0.8–1.0) + flip + JPEG-compression (q 40–95) + blur augmentation · label smoothing 0.1 · AdamW 1e-4, cosine schedule · early stopping on **validation loss** (patience 2) · class weights if imbalance > 60/40. Test-time augmentation: 5 views averaged (optional flag).
Test splits: WildDeepfake test (in-domain), **FaceForensics++ c23** and **Celeb-DF v2** (cross-dataset). Optional **GenImage** subset for non-face generated images.

---

## 6. Fusion and verifier (`agents/verifier_agent.py`)

```
usable   = tools where error == False and confidence >= 0.15
weight_i = w_base[tool] × length_factor_i × confidence_i
fused    = Σ weight_i · score_i / Σ weight_i               # w_base from validation AUROC, fixed in config
spread   = max(score) − min(score) over usable tools
```

- **Verdict:** `fused ≥ T_hi` → synthetic; `fused ≤ T_lo` → authentic; otherwise uncertain. `T_lo` and `T_hi` are chosen on the **validation** PR curve (not 0.5) and stored in `config`.
- **Confidence:** `abs(fused − 0.5)·2`, reduced by 0.15 if `spread > 0.35`, and capped at 0.95.
- **Reflexion (max 1 retry):** if `spread > 0.35` and `retries == 0`, re-run the LLM/VLM judge once with the other tools' scores and the retrieved evidence in its prompt ("your peers disagree; re-examine"). Then re-fuse with `retries = 1`.
- **Escalate** if fewer than 2 usable tools, or the verdict is uncertain, or `spread > 0.35` after retry.
- **RAG never changes the score.** It grounds the report (see §7). One exception: a retrieved **verified** case neighbour with similarity ≥ 0.85 is shown in the report, and disagreement with it adds an escalate flag.

## 7. RAG (`rag/`)

- ChromaDB `PersistentClient`, collection created with `metadata={"hnsw:space": "cosine"}`; similarity = `1 − distance`.
- Embedding model fixed: `sentence-transformers/all-MiniLM-L6-v2` (CPU).
- **Layer 1 (literature):** 8–10 hand-written summaries, chunked by concept. **Layer 2 (generator fingerprints):** one note per generator family (GPT, Llama, Qwen, StyleGAN, Stable Diffusion, face-swap variants). **Layer 3 (case memory):** every persisted job is stored with `verified=false`. A human correction flips it to `verified=true` with the corrected label.
- **CRAG-lite:** drop hits with similarity < 0.5. If none remain, rewrite the query from the top tool's `raw_features` and retry once. If still none, the report says "no supporting evidence retrieved".
- **Citation-forced reports:** the report lists `citations` (source ids of retained hits) next to each claim. The verifier may not cite a source that was not retrieved.
- Layer 1 and 2 notes must be in your own words. Do not paste abstracts.

## 8. API, storage, security

| Endpoint | Purpose |
|---|---|
| `POST /analyze/text` | body `{text}`, 20–20 000 words |
| `POST /analyze/image` | multipart upload; JPEG/PNG/WebP sniffed by content, ≤ 10 MB, ≤ 4096 px |
| `GET /jobs/{id}` | stored report |
| `POST /jobs/{id}/correction` | human label → updates case memory |
| `GET /health` | liveness |

- **Config:** one `Settings` class with `extra="ignore"`, every env var declared, and `ANTHROPIC_API_KEY` required only when an LLM tool runs. It loads `.env` for all modules (no bare `os.getenv` in tools).
- **Evidence store:** `job_id, input_sha256, modality, tool_verdicts, fused_score, verdict, confidence, escalate, citations, model_versions, created_at, human_correction`.
- **Logging:** structured JSON line per node (job id, tool, score, latency). Langfuse is optional, week 16 only.
- **Docker:** `.dockerignore` excludes `.env`, `models/`, `data/`, `chroma_data/`. Models are mounted as a volume. Compose runs api + frontend, with Chroma as an in-process persistent volume. GPU is optional via compose `deploy.resources`.

---

## 9. Final repo structure

```
forensics-agent/
├── agents/
│   ├── schemas.py
│   ├── orchestrator.py
│   ├── verifier_agent.py          # fusion, thresholds, reflexion, escalation
│   ├── text_agent/
│   │   ├── agent.py
│   │   └── tools/ {dl_classifier.py, slm_binoculars.py, llm_judge.py}
│   └── image_agent/
│       ├── agent.py
│       └── tools/ {dl_classifier.py, gradcam.py, vlm_judge.py}
├── rag/ {ingest.py, retriever.py, case_memory.py, knowledge_base/}
├── api/ {main.py, schemas.py, validation.py}
├── evidence_store/ {models.py, hashing.py}
├── training/ {train_text_classifier.py, train_image_model.py, calibrate.py}
├── eval/ {run_benchmark.py, splits.py, visualize.py, results/, plots/}
├── frontend/app.py
├── tests/ {test_verifier.py, test_fusion.py, test_api.py, test_tools_parsing.py, test_rag.py}
├── config.py  ·  .env.example  ·  Dockerfile.api  ·  Dockerfile.frontend
├── docker-compose.yml  ·  .dockerignore  ·  ARCHITECTURE.md  ·  DECISIONS.md  ·  README.md
```

`training/calibrate.py` fits temperature scaling and the SLM logistic map on the validation split and writes `models/calibration.json`, which the tools load. This is the only place thresholds and base weights are set.

---

## 10. Evaluation protocol (this produces the paper)

**Splits**
- Text: train DL on HC3-train. Test on HC3-test (in-domain), on MAGE and RAID non-ChatGPT generators (cross-generator), and on RAID paraphrase/synonym-swap (adversarial).
- Image: train on WildDeepfake. Test on WildDeepfake-test (in-domain), FF++ c23 and Celeb-DF v2 (cross-dataset).
- Test sets are touched **once**, after all thresholds and weights are frozen from validation.

**Metrics:** AUROC, average precision, F1, TPR@1%FPR, ECE (calibration), escalation rate, and accuracy-vs-coverage (accuracy on non-escalated samples). 95% bootstrap CIs.
The benchmark stores `fused_score` and each tool's score per sample. All plots use `fused_score`, never `confidence`.

**Ablation rows (each a config flag, no code edits):** each tool alone · each pair · full without RAG · full with RAG · full with Reflexion.

Headline table:

| Method | In-domain | Cross-generator | Adversarial |
|---|---|---|---|
| DL only | | | |
| SLM only | | | |
| LLM only | | | |
| Full agentic + RAG | | | |

An honest negative result is acceptable. Report it either way.

---

## 11. Roadmap (16 weeks)

| Week | Deliverable | Exit test |
|---|---|---|
| 1 | **Stabilise.** Fix schema, `Settings`, frontend, upload endpoint. Remove XGBoost, GPT-2 Binoculars and SD-Turbo. Add `.dockerignore`. | `python agents/orchestrator.py` returns a full report; `pytest` green |
| 2 | Text tools 1: LLM judge (robust parsing) and Qwen Binoculars | each tool returns a valid `ToolVerdict` on sample text |
| 3–4 | Train DeBERTa on HC3. Write `calibrate.py`. Individual text benchmarks. | val AUROC recorded, calibration file written |
| 5 | Retrain EfficientNet (new recipe), add Grad-CAM, and run the image VLM judge | val loss curve no longer diverges; heatmap saved |
| 6 | Image benchmarks on WildDeepfake, FF++ and Celeb-DF individually | per-tool AUROC table |
| 7 | Fusion and verifier: weights, thresholds, Reflexion, escalation | unit tests for every branch |
| 8–9 | RAG layers 1 and 2, cosine Chroma, CRAG-lite, citation-forced report | retrieval test returns cited hits |
| 10 | RAG layer 3 (case memory) and the correction endpoint | correction changes a later report |
| 11 | Full-pipeline integration and structured logging | 100-sample smoke run with no crashes |
| 12 | Final ablation runs, text side | text rows of the table filled |
| 13 | Final ablation runs, image side | image rows of the table filled |
| 14 | Frontend polish, Docker, CI (lint + tests, no heavy installs) | `docker compose up` works from a clean clone |
| 15 | Plots, paper tables, error analysis (false positives and false negatives) | all figures reproducible from one script |
| 16 | Buffer, writeup, optional Langfuse | — |

## 12. Environment variables (all declared in `Settings`)

```
ANTHROPIC_API_KEY=
LLM_MODEL_ID=claude-sonnet-5-5
TEXT_DL_MODEL_PATH=models/deberta_text.pt
IMAGE_MODEL_PATH=models/efficientnet_b4.pt
CALIBRATION_PATH=models/calibration.json
CHROMA_PERSIST_DIR=./chroma_data
EVIDENCE_DB_URL=sqlite:///evidence_store/forensics.db
JOBS_DIR=./jobs
LOG_LEVEL=INFO
```

## 13. Future work (state in the report, don't build)

Distributed tracing (beyond optional Langfuse), continuous evaluation, legal-grade chain-of-custody, rate limiting and load testing, prompt-injection red-teaming, LoRA-fine-tuned SLM classifier, reconstruction-residual image detector.

## 14. Definition of done

- One command runs the full pipeline for text and image.
- The ablation table is filled from `eval/results/*.json` with confidence intervals.
- Every report cites retrieved sources or says none were found.
- Tests cover the verifier, fusion, parsing, validation and RAG.
- `docker compose up` works from a clean clone with no secrets baked into images.
