---
modality: text
---
# Fine-tuned transformer classifiers for AI-text detection

A common supervised approach fine-tunes an encoder such as RoBERTa or DeBERTa to classify text as
human or machine. In-domain accuracy is usually very high because the model learns generator-specific
style. The weakness is brittleness: accuracy tends to fall on unseen generators, new domains,
paraphrased text and very short passages.

Practical points used in this project: long texts are scored in overlapping token windows and the
window logits are averaged; the output is temperature-scaled on held-out data so that a score of 0.8
means roughly 80 percent of such texts are machine-written; and the classifier is always fused with
zero-shot and reasoning-based signals rather than trusted alone.
