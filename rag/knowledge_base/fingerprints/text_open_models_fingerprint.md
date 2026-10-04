---
modality: text
---
# Generator fingerprint: open models such as Llama and Qwen (text)

Open instruction-tuned models share much of the chat-model style: polished, generic and structured
output. Differences appear with decoding. Greedy or low-temperature decoding tends to produce
repetition, recycled phrasing and unusually predictable text with very low perplexity, while higher
temperature or nucleus sampling gives more varied text that looks less machine-like to perplexity
detectors. Smaller models add more factual slips and topic drift.

A detector trained on one generator family may miss another, so cross-generator testing on RAID and
MAGE matters. When a text looks unusually predictable yet lacks the signposting phrases typical of
GPT-style models, an open model with conservative decoding is a plausible source.
