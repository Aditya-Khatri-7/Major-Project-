---
modality: image
---
# Local edits, inpainting and splicing

A local alteration replaces only part of a photograph: a face region, an object, clothing or background. Generative inpainting fills a masked region with newly synthesised pixels that match the surroundings in colour and lighting but not in sensor noise or compression history.

## What reviewers look for
- A region that a detector scores as synthetic while the rest of the image scores as real (a localised hot spot on the evidence map).
- Differences in fine noise texture between the edited region and the surroundings.
- Boundary effects around the edited region: smoothing, repeated textures, inconsistent sharpness.
- Face geometry details: asymmetric ears, jewellery that merges into skin, inconsistent hair edges, irregular teeth and pupils.

## Limits
Small edits, heavy JPEG compression, resizing and filters all weaken pixel-level evidence. A detector hot spot is evidence for human review, not proof that the region was edited.
