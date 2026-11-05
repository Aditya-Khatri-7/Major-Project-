---
modality: image
---
# Modern image generators and why older detectors fail

Detectors trained on older face swaps (FaceForensics++, Celeb-DF) and early GAN images do not transfer reliably to current text-to-image systems (Flux, Midjourney, GPT-image, Stable Diffusion 3.x, Ideogram). Modern outputs show far fewer visible artefacts. Whole-image detectors trained on large, recent real-vs-generated collections (for example the OpenFake benchmark) transfer better.

## Practical guidance
- Do not rely on face-crop detectors alone for scenes created by modern generators.
- Combine independent signals: provenance metadata, a whole-image detector, and a face-specific detector.
- Disagreement between detectors should lower confidence and trigger human review.
- Detector accuracy drops after heavy compression, resizing and screenshotting.
