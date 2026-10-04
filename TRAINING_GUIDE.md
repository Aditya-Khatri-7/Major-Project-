# Training and Evaluation Guide

Everything except the actual training runs is already built and tested. This guide takes you from a clean
machine to the paper's results table. Run every command from the `forensics-agent/` folder.

Hardware used: RTX 5080 16 GB. Everything (data, models, caches) stays on **D:** inside this folder; downloads go to `hf_cache/` unless `HF_HOME` is set.

---

## 0. One-time setup

```powershell
# environment (PyTorch already installed with CUDA; keep it)
pip install -r requirements-dev.txt
copy .env.example .env          # then put your GEMINI_API_KEY (or ANTHROPIC_API_KEY) in .env

# keep model downloads off C: and make downloads tolerant of slow networks
setx HF_HOME D:\hf_cache
setx HF_HUB_DOWNLOAD_TIMEOUT 180
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
# (close and reopen the terminal so these take effect)

pytest tests -q                 # expect: all tests pass
```

**Torch note.** `microsoft/deberta-v3-small` ships only a pickle checkpoint on its main branch, and current
`transformers` refuse to load it with torch < 2.6. The training script automatically falls back to the hub's
safetensors copy (`refs/pr/4`). If that download keeps failing, upgrade torch to 2.6+.

---

## 1. Download the data (do the access requests FIRST, they take days)

| Data | Where | Needed for |
|---|---|---|
| WildDeepfake | already at `D:\archive` (train/valid/test, real/fake) | image train + in-domain test |
| FaceForensics++ c23 | https://github.com/ondyari/FaceForensics (fill the access form) | image cross-dataset test |
| Celeb-DF v2 | https://github.com/yuezunli/celeb-deepfakeforensics (request form) | image cross-dataset test |
| HC3 | downloaded automatically by the prepare script | text train / val / calib / test |
| MAGE | Hugging Face `yaful/MAGE`, downloaded by the prepare script | text cross-generator test |
| RAID (labeled train split) | `pip install raid-bench`, then save the train split to `data/raw/raid_train.csv` | text cross-generator + adversarial test |

Check the image dataset layout:

```powershell
python data/check_datasets.py image --data-dir D:\archive
```

## 2. Prepare the data

```powershell
# HC3: train / val / calib / test, split by question (no leakage)
python data/prepare_text.py hc3

# MAGE: cross-generator test (VERIFY the label direction from the printed samples; if AI/human look swapped,
# change --human-value)
python data/prepare_text.py hf --dataset yaful/MAGE --split test --text-col text --label-col label --human-value 1 --src-col src --n 3000 --name mage_test

# RAID: cross-generator and adversarial sets
python data/prepare_text.py raid --file data\raw\raid_train.csv --n 3000

python data/check_datasets.py text

# FF++ and Celeb-DF frames (face crops)
pip install facenet-pytorch --no-deps      # optional, better face detector
python data/prepare_frames.py ffpp --root D:\FaceForensics++ --methods Deepfakes FaceSwap NeuralTextures --frames-per-video 10 --max-videos 150
python data/prepare_frames.py celebdf --root D:\Celeb-DF-v2 --frames-per-video 10
```

## 3. Smoke test before the long runs (5 minutes)

```powershell
python training/train_image_model.py --data-dir D:\archive --epochs 1 --max-train-samples 400 --max-val-samples 200
python training/train_text_classifier.py --train-csv data/processed/text/hc3_train.csv --val-csv data/processed/text/hc3_val.csv --epochs 1 --max-train-samples 400 --max-val-samples 200
```
Both must finish and write curves under `eval/plots/`. Delete `models/` outputs afterwards if you like.

## 4. Train the text classifier (DeBERTa-v3-small)

```powershell
python training/train_text_classifier.py `
  --train-csv data/processed/text/hc3_train.csv --val-csv data/processed/text/hc3_val.csv `
  --epochs 20 --patience 3
```
- Up to 20 epochs. After every epoch it prints train/validation loss, accuracy, precision, recall, F1, ROC-AUC.
- Whenever validation ROC-AUC improves, the model in `models/text_dl/` is **replaced** by the new best. It stops
  after 3 epochs without improvement (so it usually ends around epoch 4-8; that is expected).
- Expect roughly 20-60 minutes per epoch on the 3050 for full HC3. If too slow add `--max-train-samples 30000`.
- Interrupted? Add `--resume` to the same command.
- Out of memory? `--batch-size 4 --grad-accum 8`. Loss NaN with bf16? `--precision fp32` (slower).
- Watch progress: `eval/plots/text_dl/training_curves.png` is redrawn every epoch. Tables: `eval/results/text_dl/history.csv`.

## 5. Train the image classifier (EfficientNet-B4)

```powershell
python training/train_image_model.py --data-dir D:\archive --epochs 20 --patience 4
```
- Same best-checkpoint and early-stopping behaviour; best weights go to `models/efficientnet_b4.pt`
  (+ `.meta.json` with the class mapping and the best epoch's metrics).
- Your previous run overfit (train 99.7% vs validation 88.9%). This version adds JPEG/blur/crop/colour augmentation,
  label smoothing and a cosine schedule. If an epoch takes too long: `--max-train-samples 60000`.
- Resume with `--resume`. Start from your old weights with `--init-weights models/efficientnet_b4_wilddeepfake.pt`.
- Only 60% of `valid/` is used for early stopping. The other 40% is reserved for calibration (step 7).

## 6. Build the knowledge base (RAG)

```powershell
python rag/ingest.py --kb-dir rag/knowledge_base
```
Then read and improve the notes in `rag/knowledge_base/` (they are drafts: check each claim against the paper
before citing it in your report).

## 7. Score the calibration split, then calibrate

`score_dataset.py` runs every tool once per sample and caches the result, so later steps cost nothing.
The LLM/VLM tools cost API money: start with `--limit 500` and raise it later (already-scored samples are skipped).

```powershell
# TEXT
python eval/score_dataset.py --modality text --data data/processed/text/hc3_calib.csv --out-dir eval/cache/text_calib --limit 1000
python training/calibrate.py --modality text --cache-dir eval/cache/text_calib

# IMAGE (the 40% of valid/ not used for early stopping)
python eval/score_dataset.py --modality image --data D:\archive\valid --subset calib --out-dir eval/cache/image_calib --limit 1000
python training/calibrate.py --modality image --cache-dir eval/cache/image_calib
```
This writes `models/calibration.json` (temperatures, Binoculars mapping, fusion weights, thresholds).
Calibration needs at least 100 usable samples per tool. **Do this before any test set.**

## 8. Score the test sets and run the benchmark (touch each test set ONCE)

```powershell
# TEXT: in-domain, cross-generator, adversarial
foreach ($t in @("hc3_test","mage_test","raid_crossgen","raid_adv")) {
  python eval/score_dataset.py --modality text --data data/processed/text/$t.csv --out-dir eval/cache/text_$t --limit 1500
  python eval/run_benchmark.py --modality text --cache-dir eval/cache/text_$t --tune-cache-dir eval/cache/text_calib --name $t --output-dir eval/results/text/$t
}

# IMAGE: in-domain, then the two cross-dataset sets
python eval/score_dataset.py --modality image --data D:\archive\test --out-dir eval/cache/image_wd_test --limit 1500
python eval/run_benchmark.py --modality image --cache-dir eval/cache/image_wd_test --tune-cache-dir eval/cache/image_calib --name wilddeepfake_test --output-dir eval/results/image/wilddeepfake_test
python eval/score_dataset.py --modality image --data data/processed/image/ffpp --out-dir eval/cache/image_ffpp --limit 1500
python eval/run_benchmark.py --modality image --cache-dir eval/cache/image_ffpp --tune-cache-dir eval/cache/image_calib --name ffpp --output-dir eval/results/image/ffpp
python eval/score_dataset.py --modality image --data data/processed/image/celebdf --out-dir eval/cache/image_celebdf --limit 1500
python eval/run_benchmark.py --modality image --cache-dir eval/cache/image_celebdf --tune-cache-dir eval/cache/image_calib --name celebdf --output-dir eval/results/image/celebdf
```
Each run prints AUROC / accuracy / F1 for every single tool, every pair, and the full system, with 95% bootstrap
confidence intervals, and saves ROC/PR overlays, confusion matrix, reliability diagram and a summary card under
`eval/results/<modality>/<name>/plots/`.

**Reflexion row (optional, extra API cost only on disagreement cases):**
```powershell
python eval/run_reflexion.py --modality text --data data/processed/text/hc3_test.csv --cache-dir eval/cache/text_hc3_test
python eval/run_benchmark.py ...same command as above...      # now also reports "full+reflexion"
```
**RAG row:** add `--rag` to `run_benchmark.py` to report citation coverage.

## 9. Build the paper tables

```powershell
python eval/make_tables.py --modality text --results eval/results/text/hc3_test/results.json eval/results/text/mage_test/results.json eval/results/text/raid_crossgen/results.json eval/results/text/raid_adv/results.json --metrics roc_auc accuracy f1
python eval/make_tables.py --modality image --results eval/results/image/wilddeepfake_test/results.json eval/results/image/ffpp/results.json eval/results/image/celebdf/results.json
```
Outputs (`eval/results/tables/`): markdown, CSV and LaTeX tables, and bar charts. This is your headline table:
in-domain vs cross-generator vs adversarial, each tool alone vs the full system.

## 10. Run the system

```powershell
uvicorn api.main:app --reload            # terminal 1
streamlit run frontend/app.py            # terminal 2
docker compose up --build                # or everything in Docker (models mounted from ./models)
```

---

## Interpreting results honestly
- A drop from in-domain to cross-dataset / adversarial is expected and is a valid finding. Report it.
- If the full system does not beat the best single tool on some column, report that too.
- Keep the tuning (calibration) data and the test data separate; the scripts enforce this when you follow the steps.
- Verdicts are probabilistic. Never present them as proof against a person (see the limits note in the knowledge base).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Text DL model not found` | finish step 4 (or set `TEXT_DL_MODEL_PATH`) |
| all tools error in `score_dataset.py` | it aborts after 8 failures and prints the reason (missing model / API key) |
| Binoculars scores look inverted (fitted slope positive) | the label direction of your CSV is wrong |
| `ANTHROPIC_API_KEY is not set` | add it to `.env`, or score only local tools with `--tools dl,slm` |
| CUDA out of memory | smaller `--batch-size`; close other GPU programs; the API serves one job at a time |
| Hugging Face timeouts | set `HF_HUB_DOWNLOAD_TIMEOUT=180` and retry (downloads resume) |

---

## Addendum (2 Oct 2026): the multi-source image recipe and updated text model

The WildDeepfake-only image model scored AUC 0.52 on FF++ because FF++ frames were full frames, not face crops, and it
had only seen one source. See `WORKLOG.md` for the full reasoning. The corrected recipe:

```powershell
# 1. face-crop FF++ and build the mixed train/valid + held-out test sets (parallel crop workers, then assemble)
python data/build_mixed_image.py --crop-only Original      # repeat for Deepfakes FaceSwap NeuralTextures Face2Face FaceShifter
python data/build_mixed_image.py --skip-crop
# 2. train on the mix
python training/train_image_model.py --data-dir data/processed/image/mixed --epochs 20 --patience 4
# 3. evaluate on held-out sets (threshold taken from validation, never from the test set)
python eval/eval_image_sets.py --name image_v3 --threshold-from data/processed/image/mixed/valid `
  --set wd_test=data/processed/image/wilddeepfake/test --set ffpp_seen=data/processed/image/tests/ffpp_seen `
  --set ffpp_unseen=data/processed/image/tests/ffpp_unseen --set celebdf=data/processed/image/tests/celebdf
```

Text: the active model is `models/text_dl_v2` (DeBERTa-v3-base on `mix_train.csv`). Use `raid_test_all`,
`raid_unseen_gen`, `raid_unseen_attack`, `mage_test` and `hc3_test` as test sets. Do not use the old
`raid_adv` / `raid_crossgen` (they overlap the v2 training mix).
