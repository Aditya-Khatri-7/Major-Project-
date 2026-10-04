# Demo checklist

## 10 minutes before
1. Open PowerShell in `D:\!Forensics agent`.
2. `python -m pytest tests -q` -> expect `79 passed`.
3. Terminal 1: `python -m uvicorn api.main:app --port 8000` (first request loads models, about 20-30 s; send one warm-up text first).
4. Terminal 2: `python -m streamlit run frontend/app.py` -> http://localhost:8501.
5. Check http://localhost:8000/health shows `text_dl_model`, `image_dl_model`, `calibration_file` all true.
   `llm_configured: false` is expected until a judge key is added; the warning banner in the app is normal.

## Order of the demo (inputs are in `demo_samples/`)
1. `text_human_written.txt` -> authentic. `text_AI_generated.txt` -> synthetic. Point at the per-tool breakdown and the cited sources.
2. `image_real.jpg` -> authentic. `image_deepfake.jpg` -> synthetic, with the Grad-CAM heatmap.
3. Paste any AI-generated illustration -> routed to the general probe (the scope guard explains why the face tools are skipped).
4. Honest weakness: upload a screenshot of the deepfake and show the confidence drop / human-review flag.

## If something breaks
* API not reachable: check Terminal 1 for the traceback; `logs/` has JSON lines per node.
* GPU busy: close other GPU programs; the pipeline runs one job at a time (a second request gets HTTP 503, not a hang).
* Anything with the judges: they are optional; leave `ENABLED_*_TOOLS` alone, the local tools carry the demo.

## After adding the Gemini key (the one remaining task)
```powershell
copy .env.example .env      # then edit: GEMINI_API_KEY=...  LLM_PROVIDER=gemini
python eval/smoke_judges.py # prints a verdict per demo input; exit code 0 = all four judge calls worked
```
Then restart the API; `/health` should show `llm_configured: true`.

## Numbers to remember
Text AUC 0.96-0.998 across five held-out sets; image AUC 0.976-0.999 on seen datasets, 0.794 on unseen manipulation methods.
The system escalates 30-52% of cases to a human and is 92-99.6% accurate on the rest. Re-uploaded images raise false alarms.
