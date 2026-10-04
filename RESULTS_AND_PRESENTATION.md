# Forensics Agent: Results Summary and Presentation Script

> Full details: `FINAL_PROJECT_REPORT.md`.

## 1. One-paragraph summary

An agentic AI + RAG system that decides whether a **text** is AI-generated or an **image** is a deepfake, explains why,
and escalates uncertain cases to a human. Several tools score each input (a trained DeBERTa text classifier, an
EfficientNet-B4 + CLIP image detector, with Gemini/Claude judges as optional extras). A verifier fuses the calibrated scores,
checks for disagreement, retrieves cited evidence from a knowledge base, and writes an audited report.

## 2. Pipeline

```
Streamlit -> FastAPI -> LangGraph
  text : DeBERTa-v3-base classifier | Qwen Binoculars | (LLM judge, needs a Gemini/Claude key)
  image: EfficientNet-B4 + Grad-CAM | CLIP ViT-L/14 probe | (vision-LLM judge, needs a key)
  -> RAG (knowledge base + case memory, cited) -> verifier (calibrated fusion, disagreement check, escalate flag)
  -> report -> evidence store (SHA-256 audit record)
```

## 3. Results (all on held-out data, leakage-checked)

### Text (DeBERTa-v3-base on a HC3 + MAGE + RAID mix, plus Binoculars; 1,000 held-out samples per set, 95% CIs)

| Test set | What it tests | AUC (full system) | TPR @ 1% false positives | Escalated | Accuracy when decided |
|---|---|---|---|---|---|
| HC3 test | in-domain | 0.998 | 1.00 | 33% | 0.996 |
| MAGE test | many generators and domains | 0.961 | 0.58 | 52% | 0.958 |
| RAID test documents | all generators, all attacks | 0.985 | 0.81 | 30% | 0.966 |
| RAID unseen generator (cohere) | never trained on | 0.957 | 0.60 | 43% | 0.933 |
| RAID unseen attacks | paraphrase / homoglyph / synonym | 0.962 | 0.61 | 36% | 0.920 |

Before the mixed training the same kind of model scored MAGE 0.658 and unseen-generator 0.809. After adding input normalisation
(zero-width and look-alike characters), the homoglyph attack improved from AUC 0.72 to 0.99 without retraining.

### Image (EfficientNet-B4 on a three-dataset mix, plus CLIP probe; 1,500 held-out images per set)

| Test set | What it tests | AUC (full system) | TPR @ 1% false positives | Escalated | Accuracy when decided |
|---|---|---|---|---|---|
| WildDeepfake test | in-domain | 0.977 | 0.67 | 29% | 0.977 |
| FF++ (seen methods, unseen videos) | other dataset | 0.976 | 0.70 | 39% | 0.954 |
| Celeb-DF (official Test split) | other dataset | 0.999 | 0.99 | 29% | 0.995 |
| FF++ unseen methods (Face2Face, FaceShifter) | methods never trained on | 0.794 | 0.13 | 53% | 0.759 |

Single-source baseline (WildDeepfake only): FF++ 0.52 (chance), Celeb-DF 0.69.
Re-upload test: AUC mostly holds after JPEG / downscale / screenshot-like damage, but false alarms on real faces rise (Celeb-DF 3% -> 9%,
WildDeepfake 7% -> 17%, FF++ 15% -> 49%).

### End-to-end check (local tools only, no API key)
- API demo: 17 of 20 texts from an unseen generator correct. Two of the three misses were flagged "uncertain" and
  escalated to a human. All 10 human texts were classified correctly.
- Real image classified authentic (0.95), deepfake classified synthetic (0.67), with Grad-CAM and an escalation flag.

### Image routing (added after live testing)
Live testing with an AI-generated illustration (a stylised neon head) exposed a scope problem: the face-deepfake tools
were trained only on face-swap deepfakes, so they called an AI illustration "authentic". Fix:

| Input type | Handled by | Held-out result |
|---|---|---|
| Face photo / crop (face-swap deepfakes) | EfficientNet-B4 + CLIP probe (existing) | see table above |
| Non-face image (illustration, AI art, objects) | NEW general probe: frozen CLIP ViT-L/14 + logistic regression, trained on DeepDetect-2025 + a real-vs-AI art set | AI-art test AUC 0.999, DeepDetect test AUC 0.968; live API check 39/40 art images correct |
| AI-synthesised faces (GAN / diffusion portraits) | face tools with the unified CLIP face probe | 63% caught at 7% false alarms offline (was 21%); still a limitation |

A zero-shot CLIP scope guard ("is this a face photo?") routes each image. The general probe was also tested on face-swap
deepfakes and is weak there (AUC 0.59-0.65), which is why faces are not sent to it.

## 4. Problems found and fixed (the research story)

| Problem | Cause | Fix | Effect |
|---|---|---|---|
| Text collapsed on MAGE (AUC 0.66) | trained on one source (HC3) | mixed HC3 + MAGE + RAID, held-out generators and attacks | 0.97 |
| Image failed on FF++ (AUC 0.52) | preprocessing mismatch: full video frames vs face crops, and single-source training | face-crop FF++, multi-source mix, stronger augmentation | 0.98 |
| Homoglyph attack broke text model (0.72) | look-alike characters break the tokenizer | input normalisation | 0.99 |
| Possible data leakage | frames/texts from one video or document in train and test | split by video / by source document, automatic leakage checks | clean tests |
| Illustration called "authentic" | no tool covered non-face images | scope guard + general AI-image probe | 39/40 on AI art |
| NaN training crashes | fp16 default and fp16 model loading | bf16 and fp32 weights | stable training |

## 5. Honest limitations (say these before they are asked)
- **Unseen deepfake methods:** 0.79 AUC. Self-blended images, extra manipulation types and CLIP were all tried;
  only CLIP fusion helped, modestly.
- **Unseen text attacks:** now AUC 0.974 (accuracy 0.925) after input normalisation; paraphrase is still the hardest case.
- **AI-synthesised faces** (GAN/diffusion portraits): improved with a unified CLIP face probe (21% -> 63% caught at 7% false
  alarms offline) but not solved.
- **The art result (AUC 0.999) is in-domain** for that dataset; unseen generators will score lower.
- **Celeb-DF and FF++ were part of training** (video-disjoint test), so those scores are per-dataset, not zero-shot.
- **Judges are implemented but not benchmarked**: the Gemini/Claude judge code is tested with mocked responses; Reflexion and the
  judge ablation rows need a key and one run of `eval/smoke_judges.py`.
- **Re-uploaded images** (screenshots, social media) raise false alarms on real faces; ranking quality mostly survives.
- **Unseen deepfake methods catch only 13% of fakes at 1% false positives.**
- Detection is probabilistic, not proof. Compression and adversarial edits reduce accuracy.

## 6. Presentation script (about 7 minutes)

**1. Problem (30 s).** AI text and deepfakes are hard to tell from real content, and a single yes/no detector is not
trustworthy evidence. We built a system that gives a verdict, shows its reasoning, and hands doubtful cases to a person.

**2. Architecture (60 s).** Show the pipeline diagram. Several independent tools score the input. A verifier combines the
calibrated scores, notices when tools disagree, and attaches cited evidence from a knowledge base. Every job is stored
with a SHA-256 hash of the input as an audit record.

**3. The key finding (90 s).** "Our first models scored 0.97-1.0 on their own training data, and failed on anything new:
text dropped to 0.66, images to chance. In-domain scores hide this."
Explain: models learned shortcuts, such as ChatGPT's tone or one dataset's compression. Two fixes: mix many sources, and
hold out whole generators, attacks and datasets for testing.

**4. Results (90 s).** Show the two result tables. Text reaches 0.96 on unseen generators. Images reach 0.98-0.999
on other datasets. Also say what the system does when unsure: it escalates 30-50% of cases and is 92-99% accurate on the rest. Mention the FF++ story: the cause was a preprocessing mismatch we found by inspecting the data, not
just a weaker model.

**5. Live demo (120 s).** Open http://localhost:8501. Use files in `demo_samples/`:
1. Paste `text_human_written.txt`, then `text_AI_generated.txt`. Point at the verdict, the per-tool breakdown and the
   cited evidence.
2. Upload `image_deepfake.jpg`. Show the Grad-CAM heatmap and the two image tools agreeing. Then upload a heavily compressed or screenshot copy to show the honest weakness (more false alarms, human review recommended).
3. Copy any image or screenshot, click **Paste image from clipboard**, and analyse it.

**6. Limitations and next steps (60 s).** Use section 5. Next: enable the Gemini judges with an API key, run the full
benchmark and ablations, and improve generalisation to unseen manipulation methods.

## 7. Likely questions

- **"Why not just report 99%?"** Because that number is in-domain. We test on generators, attacks and datasets the model
  never saw, and we report those.
- **"Is the test set clean?"** Yes. Text is split by source document, images by video, and leakage checks run
  automatically. Old test files that overlapped training were retired.
- **"What does the LLM judge add?"** It is built in as an extra tool (Gemini or Claude, switchable) with Reflexion, but not benchmarked
  yet because the key is added last. The system runs fully on the local tools.
- **"What happens when it is unsure?"** It reports "uncertain" or flags the case for human review instead of guessing.

## 8. Run the demo

```powershell
python -m uvicorn api.main:app --port 8000
python -m streamlit run frontend/app.py
```
Open http://localhost:8501. Demo inputs are in `demo_samples/`.
