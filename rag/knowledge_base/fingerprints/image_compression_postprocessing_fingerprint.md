---
modality: image
---
# Confounder: compression and post-processing (images)

JPEG compression, resizing, screenshots and social-media re-encoding wipe out much of the
high-frequency information that artifact detectors rely on. Real photos that were repeatedly
re-shared can look "generated" to a detector trained on clean data, and fakes can be laundered the
same way.

When an image is small, heavily compressed or clearly a screenshot, lower trust in pixel-level tools,
lean on semantic reasoning, and prefer escalation to a human when the tools disagree.
