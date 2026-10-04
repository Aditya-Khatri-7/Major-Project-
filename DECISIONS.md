# Decision log

Changes to the locked design in ARCHITECTURE.md, and why. Add a line here whenever you change a decision.

| Decision | Choice | Why |
|---|---|---|
| Checkpoint / early-stopping metric | Validation **ROC-AUC** (configurable `--monitor`), patience 3 (text) / 4 (image), max 20 epochs | You asked for "keep improving until no better result". AUC is threshold-free and robust; loss can rise while ranking still improves (your first image run). Temperature scaling repairs calibration afterwards. |
| Calibration data | Text: dedicated `calib` split of HC3. Image: deterministic 40% of `valid/` never used for early stopping | Early-stopping data must not also tune thresholds, weights and temperatures. |
| Score caching | Every tool runs once per sample; ablations, fusion and thresholds computed offline | Cuts API cost and makes all ablation rows consistent. |
| Threshold band | `t_lo`/`t_hi` from precision/NPV targets, at least +/-0.05 around the F1-optimal cut | Guarantees borderline scores are flagged even if validation data separates perfectly. |
| Text DL weights format | HF directory `models/text_dl/` (`TEXT_DL_MODEL_PATH`) instead of a single `.pt` | Saves tokenizer and label map with the model. |
| Image weights | `models/efficientnet_b4.pt` + `.meta.json`; loader falls back to the legacy `efficientnet_b4_wilddeepfake.pt` | Old weights keep working (class order fake=0, real=1). |
| DeBERTa loading | Fallback to `refs/pr/4` safetensors when the main-branch pickle cannot be loaded (torch < 2.6) | Avoids forcing a torch upgrade. |
| RAG effect on the verdict | None on the score; a human-verified contradicting neighbour (similarity >= 0.85) forces escalation | Keeps the ablation clean; RAG grounds and audits, it does not silently move scores. |
| Reflexion | At most one judge re-run, only if two or more tools are usable and disagree (spread > 0.35) and a working judge plus API key exist | Bounded cost, demonstrable loop. |
| Chroma telemetry | Disabled | No data leaves the machine. |
| Case memory content | Verdict and tool explanations only, never the analysed text | Privacy. |
| Text DL model (changed) | **DeBERTa-v3-base** trained on `mix_train.csv` (HC3 24k + MAGE 60k + RAID 52k) at `models/text_dl_v2` | v1 (v3-small, HC3 only) memorised HC3 style: MAGE AUC 0.66. v2: MAGE 0.97, RAID unseen generator 0.96. Hardware is an RTX 5080 16 GB, so the base model (~1.5 GB inference) fits easily. |
| Text input normalisation | NFKC + zero-width strip + confusable->ASCII (only for mostly-Latin text) before tokenizing | RAID homoglyph attack: AUC 0.72 -> 0.99 with no retraining. Tamper counts are surfaced as evidence. |
| Text test sets | Use `raid_test_all`, `raid_unseen_gen`, `raid_unseen_attack`, `mage_test`, `hc3_test` | Old `raid_adv` / `raid_crossgen` overlap the v2 training mix (361 / 248 texts). |
| Image training data | Multi-source mix: WildDeepfake + FF++ (Deepfakes, FaceSwap, NeuralTextures) + Celeb-DF official Train | Single-source training gave FF++ AUC 0.52. FF++ frames are now face-cropped like the other sets. Test sets are video-disjoint; Face2Face + FaceShifter are never trained on (unseen-manipulation test). |
