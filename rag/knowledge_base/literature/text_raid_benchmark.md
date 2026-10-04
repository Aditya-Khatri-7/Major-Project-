---
modality: text
---
# RAID: robustness benchmark for machine-generated text detectors

RAID (Dugan et al., 2024) is a large benchmark that covers many generators, several domains,
different decoding strategies and repetition penalties, and a set of adversarial attacks such as
paraphrasing, synonym swapping and homoglyph substitution. Its main lesson is that detectors which
look excellent on a single generator often fail on unseen generators, on unusual decoding settings,
or after light adversarial editing.

For this project RAID supplies the cross-generator and adversarial test sets. Report results per
generator and per attack, because a single average hides the failures. Detector thresholds should
be set on held-out data and then frozen before running RAID.
