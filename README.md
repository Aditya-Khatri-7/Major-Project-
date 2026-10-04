# Forensics Agent

Agentic AI + RAG system that detects AI-generated **text** and **deepfake images**, explains its reasoning,
and escalates uncertain cases to a human.

```
Client (Streamlit) -> FastAPI -> LangGraph
   text:  DeBERTa-v3 classifier | Qwen Binoculars (SLM) | LLM judge (Gemini or Claude)
   image: scope guard -> face photo: EfficientNet-B4 + Grad-CAM, CLIP probe | other images: general CLIP probe
          (+ vision-LLM judge)
   -> RAG (knowledge base + case memory, cited) -> verifier (calibrated fusion, disagreement check,
      one Reflexion retry, escalation) -> report -> evidence store (SHA-256 audit record)
```

Design decisions: [ARCHITECTURE.md](ARCHITECTURE.md), [DECISIONS.md](DECISIONS.md).
**Training, calibration and evaluation: [TRAINING_GUIDE.md](TRAINING_GUIDE.md).**

## Quick start

```bash
pip install -r requirements.txt            # PyTorch first for your CUDA version, see requirements.txt
cp .env.example .env                       # add GEMINI_API_KEY (or ANTHROPIC_API_KEY); judges are optional
python rag/ingest.py --kb-dir rag/knowledge_base
uvicorn api.main:app --reload              # http://localhost:8000/docs
# web interface: http://localhost:8000/  (served by the API itself from web/)
streamlit run frontend/app.py              # optional legacy Streamlit client, http://localhost:8501
```
The local models (text classifier, image classifier, calibration) come from the training guide; until they exist the
corresponding tools report an error verdict and the pipeline continues with the remaining tools.

## API

| Endpoint | Purpose |
|---|---|
| `POST /analyze/text` | `{"text": "..."}` (20 to 20,000 words) |
| `POST /analyze/image` | multipart upload; JPEG/PNG/WebP checked by content, max 10 MB |
| `GET /jobs/{id}` | stored audit record and report |
| `GET /jobs/{id}/gradcam` | Grad-CAM heatmap for an image job |
| `POST /jobs/{id}/correction` | `{"label": "authentic"\|"synthetic"}`: human label, feeds case memory |
| `GET /health` | liveness and which models are present |

## Docker

```bash
docker compose up --build                                          # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up  # NVIDIA GPU
```
Models are mounted read-only from `./models`; secrets come from `.env` at run time (never baked into images).

## Layout

| Folder | Contents |
|---|---|
| `api/` | HTTP layer and input validation only |
| `agents/` | orchestrator, text/image agents and tools, fusion, verifier, Reflexion, report |
| `rag/` | ChromaDB store, ingestion, retriever, case memory, `knowledge_base/` notes |
| `evidence_store/` | SQLAlchemy audit records, hashing |
| `training/` | text and image training (early stopping, best checkpoint), calibration |
| `eval/` | score cache, offline benchmark and ablations, plots, paper tables |
| `data/` | dataset preparation and checks |
| `frontend/` | Streamlit app |
| `tests/` | unit and integration tests (no models or API key needed) |

## Tests and lint

```bash
pytest tests -q
ruff check .
```

## Limits (be honest in the report)
Detection is probabilistic and degrades on unseen generators, heavy compression and adversarial edits. Not in scope:
distributed tracing, continuous evaluation, legal chain-of-custody, load testing,
prompt-injection red-teaming beyond the input-as-data hardening. Basic API protection (optional API key, rate limit,
upload retention) is built in; see `.env.example`.
