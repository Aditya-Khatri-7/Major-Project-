---
modality: text
---
# Evasion patterns: paraphrasing and humaniser tools (text)

Adversarial edits try to hide machine origin. Paraphrasing rewrites sentences with another model,
synonym substitution swaps words, and "humaniser" tools inject irregular phrasing, typos or unusual
punctuation. Character-level tricks such as homoglyphs or invisible characters also exist.

Such edits usually raise perplexity and break stylometric cues, so perplexity-based and
classifier-based detectors lose accuracy. Signs of paraphrasing include awkward synonym choices,
slightly unnatural collocations, and meaning preserved with oddly reworded structure. Reasoning-based
judges are somewhat more robust because they look at meaning and coherence, but they are not immune.
Disagreement between a perplexity tool and a judge on a passage is a useful trigger for escalation.
