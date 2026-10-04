# Work log (auto mode)

Every step: what, why, result. Newest at the bottom.

## 2 Oct 2026 — Issue fixing session

### Step 1 — Diagnose image generalisation (issues #1, #2)
- **Finding:** `data/processed/image/ffpp/*` are full 1280x720 video frames, NOT face crops. WildDeepfake (train data) and
  Celeb-DF are tight face crops. The model squashes a whole frame to 380x380 so the face is tiny -> FF++ AUC 0.52 (chance).
  This is a *preprocessing mismatch* (domain shift), not only model weakness.
- **Fix plan:** (a) crop faces from FF++ frames (margin 0.3, same as WildDeepfake style), (b) build a mixed training set
  with video/identity-disjoint splits so FF++ and Celeb-DF test sets stay unseen, (c) retrain EfficientNet with stronger
  augmentation, (d) re-evaluate on held-out sets, report honestly.
- Celeb-DF raw already has official Train/Val/Test splits (256x256 face crops) -> use them directly.
- FF++ raw frames also include Face2Face and FaceShifter, which we do NOT train on -> an *unseen-manipulation* test.

### Step 2 — Text audit (issues #3, #4, #5, #6)
- **#4 provenance solved:** v2 was trained on `mix_train.csv` (136k rows: HC3 24k + MAGE 60k + RAID 52k, built by
  `data/prepare_text_mix.py`), validated on `mix_val`. This is why v2 generalises (MAGE AUC 0.66 -> 0.97).
- **#3 leakage ruled out:** zero text overlap between mix_train and every test set (hc3_test, mage_test, raid_test_all,
  raid_unseen_gen, raid_unseen_attack); HC3 groups (questions) disjoint across splits; held-out generator (cohere) and
  held-out attacks (synonym, homoglyph, paraphrase) never appear in training. HC3 AUC 1.000 is therefore an easy
  in-domain set, not a bug.
- **Contamination note:** old `raid_adv.csv` / `raid_crossgen.csv` overlap mix_train (361 / 248 texts) -> do NOT use them
  for v2; the new `raid_*` splits are the clean ones.
- **#6 root cause found:** per-attack AUC of v2 on unseen attacks: synonym 0.968, paraphrase 0.951, **homoglyph 0.721**.
  Homoglyphs (Cyrillic/Greek look-alikes) break the tokenizer.
- **Fix:** `agents/text_agent/normalize.py` (zero-width strip, NFKC, confusable->ASCII on mostly-Latin text), applied in
  `dl_classifier.window_logits`; tamper stats (homoglyph_chars, zero_width_chars, tamper_ratio) go to `raw_features` as
  evidence. **Result: homoglyph AUC 0.721 -> 0.991, no retraining.** (synonym 0.973, paraphrase 0.958.)

### Step 3 — Housekeeping (#5, #8)
- `config.py` / `.env.example` now point to `models/text_dl_v2`. Hardware is RTX 5080 16 GB (not the 3050 6 GB in the docs),
  so DeBERTa-v3-base is within budget; recorded in DECISIONS.md.
- `TRAINING_GUIDE.md`: stale `D:\archive` paths replaced; added the multi-source image recipe addendum.
- No `.env` / API key exists -> Claude judge tools (LLM/VLM) cannot run. Pipeline runs with local tools only (`--tools dl,slm`);
  this is a standing limitation until a key is added.

### Step 4 — Image fix in progress
- Wrote `data/build_mixed_image.py` (face-crop FF++ with MTCNN, id-bucket splits, Celeb-DF official splits, unseen
  Face2Face/FaceShifter test), `eval/eval_image_sets.py` (held-out eval with validation-derived threshold).
- Strengthened augmentation in `training/train_image_model.py`: random downscale, JPEG p=0.6, wider crop, hue/gray, RandomErasing.
- Cropping 30k FF++ frames is slow with 1 process (~5 img/s) -> restarted as 6 parallel workers.

### Step 5 — Mixed dataset built, training started
- Crops done (5000/method). Mix: train 45.4k (WD 20k, FF++ 13.3k, Celeb-DF 12k), valid 9.1k; test sets: ffpp_seen 3.5k, ffpp_unseen 1.56k, celebdf 6k.
- Old WD-only model backed up to models/_v2_wd_only/. Training: batch 24, lr 1e-4, patience 4, class weights auto (FF++ is fake-heavy).

### Step 6 — Tests, RAG, pipeline prep (#7)
- pytest: 2 RAG tests failed because the test embedding double lacked chromadb 1.x methods (`embed_query`...). Fixed in
  `tests/conftest.py`; added `tests/test_normalize.py`. Suite green (63 tests).
- RAG knowledge base ingested: 21 chunks (literature + generator fingerprints) in `chroma_data`.

### Step 7 — Text calibration (local tools only)
- Scored 1500 `mix_calib` rows with DL (v2) + Binoculars; `training/calibrate.py` wrote `models/calibration.json`.
- Calibration-split AUROC: DL 0.995, Binoculars 0.756, fused 0.989 (fusion weights 0.989 / 0.512; thresholds t_lo 0.61, t_hi 0.71).
- **Honest finding:** the weak zero-shot Binoculars slightly *lowers* the fused score vs DL alone. Expected; the Claude judge
  (needs API key) is the tool that should add value. Report as is, do not hide.
- Image multi-source training running; epoch 1 val AUC 0.940 (WD-only run was 0.953 on easier single-source val).

### Step 8 — Image training stopped at plateau
- Epochs 1-12 done. Val AUC: ep1 0.940 -> ep7 0.980 -> ep11 **0.9818 (best, saved)** -> ep12 0.9815. Train acc ~0.95 vs val ~0.94
  (WD-only run: 99% vs 93%) => overfitting much reduced.
- Epoch 13 stalled on data loading (GPU 0%, 3 s/batch). Gains had flattened (<0.001 AUC), so I stopped the run manually and
  use the epoch-11 best weights (`models/efficientnet_b4_mix.pt`). Note: early stopping did not trigger; this was a manual stop.

### Step 9 — Image held-out results (issue #1, #2 solved)
| set | AUC before (WD-only) | AUC now (multi-source) |
|---|---|---|
| WildDeepfake test | 0.970 | 0.974 |
| FF++ (seen manipulations, unseen videos) | 0.518 | **0.978** |
| Celeb-DF (official Test split) | 0.693 | **0.998** |
| FF++ unseen manipulations (Face2Face, FaceShifter) | n/a | 0.755 |
- Caveat (honest): Celeb-DF and FF++ were *in the training mix* this time (video-disjoint test), so these are
  in-distribution-per-dataset scores, not zero-shot cross-dataset. The only true zero-shot test is Face2Face/FaceShifter:
  **0.755 AUC (acc 0.64)** -> the model still generalises poorly to manipulation methods it never saw. Reported as a limitation.
- Root causes fixed: (1) FF++ full frames vs face crops, (2) single-source training. Threshold 0.35 taken from validation.
- `models/efficientnet_b4.pt` now = multi-source model (old WD-only kept in models/_v2_wd_only/).

### Step 10 — Image calibration + end-to-end smoke test (#7)
- Scored 2000 `mixed/valid` calib images, calibrated: image DL AUROC 0.979, thresholds t_lo 0.25 / t_hi 0.69.
  `models/calibration.json` now holds text + image calibration.
- Smoke test (LangGraph pipeline, local tools only, no API key): 6 text samples (unseen generator) -> 6/6 correct
  (4 authentic, 2 synthetic... 2 AI correctly synthetic, 4 human correctly authentic); 6 FF++ images -> 3/3 fake detected,
  real: 1 authentic + 2 uncertain (conservatively escalated). Every report had RAG citations. Escalation on images is always
  on because only 1 tool is usable without the Claude vision judge (rule: <2 usable tools -> escalate).

## Remaining (needs the user)
1. **ANTHROPIC_API_KEY** -> enables LLM/VLM judges, Reflexion, and the full ablation table. No `.env` exists.
2. **Unseen-manipulation image generalisation** (Face2Face/FaceShifter AUC 0.755): needs more manipulation families
   (e.g. add FF++ Face2Face/FaceShifter to training and test on a different family) or a frequency-domain detector.
3. Full test-set benchmark (`eval/run_benchmark.py`) once the judges are enabled; text unseen-attack accuracy (0.73 @0.5)
   should be re-measured with normalisation on a threshold chosen from validation.

## Session 2 — Unseen-manipulation generalisation (Face2Face / FaceShifter AUC 0.755)
### Step 11 — Self-blended images (SBI)
- **Why it fails:** the model learned the artefacts of Deepfakes/FaceSwap/NeuralTextures. Other methods leave other
  artefacts. What *all* face-swap methods share is a blending trace (edge / colour / sharpness mismatch).
- **Method:** `SelfBlend` + `SBIFolder` in `training/train_image_model.py` (flag `--sbi-prob`): a REAL face is blended with a
  slightly altered copy of itself through a soft elliptical mask and labelled fake. Method-agnostic by construction.
- Face2Face + FaceShifter remain never-trained and are touched only for final evaluation (no selection on them).
- Visual check (`tmp_scripts/sbi_demo.png`): subtle colour/sharpness/edge mismatch around the face as intended.
- Run: fine-tune from `efficientnet_b4_mix.pt`, SBI p=0.3, lr 5e-5, up to 8 epochs, patience 3 -> `models/efficientnet_b4_sbi.pt`.

### Step 12 — SBI result: did NOT help (negative result, kept honest)
- SBI fine-tune stopped after epoch 4 (val AUC plateau 0.964 < baseline 0.982). Epoch-3 weights on held-out sets:
  ffpp_unseen AUC **0.738** (baseline 0.755), wd_test 0.958, ffpp_seen 0.967, celebdf 0.997 -> slightly WORSE everywhere.
- Per-method (both models): Face2Face 0.77 / 0.73, FaceShifter 0.74 / 0.75; ~50% of fakes missed, FPR 12-15% on real frames.
- Interpretation: SBI assumes a blended face swap; Face2Face is reenactment (no seam). Self-blending adds no transferable cue here.
- Mistake owned: I deleted `ffpp_crops` earlier (cache). Only test-id crops survive (hard links) -> re-cropping Face2Face /
  FaceShifter for train ids (<860) to run a leave-one-out diversity experiment.
### Step 13 — Next approach: manipulation diversity, tested leave-one-out
- Run A: add Face2Face to training, test on FaceShifter (never trained). Run B: add FaceShifter, test on Face2Face.
- If diversity transfers (AUC up on the held-out method), final model trains on all 5 FF++ methods and the LOO numbers are
  the honest zero-shot estimate.
- Built mixed_plus_Face2Face / mixed_plus_FaceShifter (+3670 train/valid frames each, no test ids). Launched runs A (+F2F) and B (+FS): fine-tune from mix weights, lr 5e-5, 4 epochs, patience 2.
- Lesson: running two B4 trainings at once filled the 16 GB VRAM (15.99 GB) and slowed both ~20x. Stopped B, queued it after A (sequential).

### Step 14 — Leave-one-out result: diversity does NOT transfer (negative result)
- Run A (mix + Face2Face train frames): Face2Face (now seen method, unseen videos) AUC **0.975**; FaceShifter (never seen) **0.71**
  (baseline 0.74). Run B (+FaceShifter) was stopped to free the GPU; A already shows the effect.
- Conclusion: a fine-tuned CNN learns per-method fingerprints; adding one method helps *that* method only.
### Step 15 — Next: frozen CLIP ViT-L/14 + linear probe (known to generalise to unseen generators)
- Plan: extract CLIP image embeddings, fit logistic regression on the seen mix only, test zero-shot on Face2Face/FaceShifter.
- If it generalises, add it as a second image tool (`clip_probe`) and fuse with EfficientNet.

### Step 16 — CLIP probe + ensemble (3 Oct)
- CLIP ViT-L/14 frozen + logistic probe (fit on training mix only, C chosen on valid): wd_test 0.935, ffpp_seen 0.825,
  **ffpp_unseen 0.722**, celebdf 0.914 -> alone it is NOT better than the CNN on unseen methods.
- **Ensemble (mean of probabilities, same images):** ffpp_unseen CNN 0.755 -> **0.798**; wd_test 0.975 -> 0.982;
  ffpp_seen 0.977 -> 0.966; celebdf 0.999 -> 0.998. Errors are complementary -> real but modest (+0.04) gain.
- Honest conclusion so far: zero-shot cross-manipulation AUC on c32-compressed FF++ frames stays ~0.75-0.80 with every
  method tried (SBI, more manipulation methods, CLIP). Not "solved"; improved to ~0.80 and documented as a limitation.
- Decision: integrate the CLIP probe as a second image tool (`clip_probe`) so fusion/verifier can use it.

### Step 17 — CLIP integrated as `image_clip` tool; full-system benchmark (3 Oct)
- New tool `agents/image_agent/tools/clip_probe.py`; registered in `common.py`, `calibration.py`, `image_agent/agent.py`,
  `score_dataset.py`, `calibrate.py` (temperature per image tool). Default `ENABLED_IMAGE_TOOLS=dl,clip,vlm`.
- Calibration split (2000 imgs): CLIP AUROC 0.911, CNN 0.979, fused 0.980; thresholds t_lo 0.35 / t_hi 0.45.
- Benchmark through the pipeline harness (1500 samples/set, thresholds from calibration split, not the test set):
  | set | CNN | CLIP | full (fused) |
  |---|---|---|---|
  | ffpp_unseen (Face2Face/FaceShifter) | 0.753 | 0.721 | **0.789** |
  | ffpp_seen | 0.980 | 0.828 | 0.978 |
  | celebdf | 0.999 | 0.924 | 0.998 |
- Net: unseen-manipulation AUC 0.755 -> ~0.79 (+0.035) at a cost of <=0.003 AUC on seen sets. Zero-shot generalisation is
  improved but NOT solved; remains a documented limitation. The Claude VLM judge (needs API key) is the remaining lever.

## Session 3 (4 Oct 2026): audit and hardening
- Put the project under git; pushed to `main` and `Aditya-Khatri` of github.com/Aditya-Khatri-7/Major-Project-. Large artefacts (data, models, caches) stay local.
- Image calibration was stale (fitted to the retired CLIP probe). Re-scored the 2000-image calibration split and the four held-out sets, refit
  (thresholds 0.35 / 0.50), reran benchmarks. Fused AUC: WD 0.977, FF++ seen 0.976, Celeb-DF 0.999, FF++ unseen 0.794.
- Text: scored five held-out sets (1000 each) with the real tools. Binoculars never improved fused AUC (-0.001 to -0.006), so its weight is
  capped at 0.10 (`--weight-cap`). Found that the weak tool still triggered the disagreement rule (75% escalation on MAGE); tools with weight
  < 0.3 are now advisory -> escalation 30-52%, accuracy on decided cases 92-99.6%.
- Verdict logic: non-face images use the general probe as the designated sole detector (no forced escalation when decisive).
- Added a Gemini judge adapter (REST/httpx, `LLM_PROVIDER`), not exercised with a real key yet (`eval/smoke_judges.py`).
- API: optional key, rate limit, 503 on busy queue, opt-in retention, knowledge base auto-ingest. Requirements fixed (scikit-learn/joblib/httpx were
  missing from runtime deps), compose points at text_dl_v2, CI now builds both Docker images. Lint clean; 79 tests.
- Re-upload robustness test: AUC mostly survives, false alarms on real faces grow (FF++ up to 49%). Reported as a limitation.
- Moved old checkpoints/experiments to `models/_archive/` (see docs/MODELS.md). Note: I previously called the duplicate homoglyph keys a bug; both
  repeats mapped to the same value, so it was harmless lint.
- Gemini key added (4 Oct, evening). `eval/smoke_judges.py`: all four judge calls valid, but the vision judge scored the demo deepfake 0.05 (wrong, confidence 0.85). Calibration scoring then hit the free-tier cap (20 requests/day/model) and logged 101 quota errors; those records were moved to `eval/cache/_old/`, not used. Added fail-fast quota handling and conservative judge weights (0.3).
- New web interface in `web/` served at `/` (replaces Streamlit for the demo). Tests now ignore the local `.env`.
- Ran the real API end-to-end with the demo samples: 4/4 correct, citations present, Grad-CAM served.
