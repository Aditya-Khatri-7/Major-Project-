---
modality: general
---
# Limits of AI-content detection and bias risks

Detection is probabilistic, not proof. Detectors make both false positives (flagging genuine content)
and false negatives, and their error rates depend on the generator, the domain, editing and
compression. Liang et al. (2023) showed that several text detectors wrongly flag writing by
non-native English speakers as machine-generated, because plain, low-perplexity prose looks
machine-like to perplexity-based methods.

For this reason the system reports calibrated confidence, exposes each tool's evidence, escalates
uncertain or contested cases to a human, and never presents a verdict as conclusive evidence
against a person.
