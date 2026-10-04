---
modality: text
---
# DetectGPT and probability-curvature detection

DetectGPT (Mitchell et al., 2023, arXiv:2301.11305) is built on the observation that text sampled
from a language model tends to sit near a local maximum of that model's log-probability. Small
perturbations of machine text usually lower its log-probability more than perturbations of human
text do. The detector compares the original text with many perturbed copies.

It works zero-shot but is expensive, because each decision needs many perturbation and scoring
passes, and it assumes access to the source model's probabilities. Binoculars later achieved a
similar zero-shot effect with two forward passes instead of many, which is why this project uses
the cheaper cross-perplexity idea.
