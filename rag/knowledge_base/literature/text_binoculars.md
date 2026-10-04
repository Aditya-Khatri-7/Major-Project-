---
modality: text
---
# Binoculars: zero-shot detection with two language models

Binoculars (Hans et al., 2024, arXiv:2401.12070) detects machine-generated text without training on
any generator. It runs two closely related language models, an observer and a performer, over the
same text. The score is the text's log-perplexity under one model divided by the cross-perplexity,
which measures how surprising one model's next-token predictions are to the other.

Dividing by cross-perplexity normalises for how predictable the prompt or topic already is, so a
plain low-perplexity test is no longer fooled by easy text. Low scores indicate machine-like text.
Strengths: needs no target-generator data and generalises across model families. Limits: unreliable
on short passages (fewer than roughly 50 words), needs a threshold calibrated per model pair, and a
small model pair is weaker than the large pair used in the paper.
