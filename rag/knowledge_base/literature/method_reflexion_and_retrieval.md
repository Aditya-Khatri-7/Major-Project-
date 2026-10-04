---
modality: general
---
# Reflexion, Self-RAG and corrective retrieval

Reflexion (Shinn et al., 2023) lets an agent improve by reflecting in language on a failed or
doubtful attempt and trying again with that reflection as context. Self-RAG (Asai et al., 2023) and
Corrective RAG (Yan et al., 2024) add self-critique to retrieval: judge whether retrieved passages
are relevant and, if not, re-query or discard them.

This project uses lightweight versions of both. When detectors disagree strongly, the judge model is
re-run once with the other detectors' outputs and retrieved notes as context. Retrieved passages
below a similarity threshold are dropped, and if none remain the query is rewritten once. Reports
cite only sources that were actually retrieved.
