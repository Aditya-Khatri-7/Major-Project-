---
modality: image
---
# Generator fingerprint: GAN-generated images such as StyleGAN faces

GAN images often carry structural artifacts: asymmetric or mismatched earrings and glasses, distorted
teeth, hair strands that merge into the background, strange blob-like texture patches, and background
regions that do not make geometric sense. Upsampling layers can leave periodic patterns that show up as
peaks in the frequency spectrum.

Cues can be subtle in modern high-resolution GANs, and resizing or JPEG compression weakens the spectral
evidence. Treat any single cue as weak evidence and prefer several agreeing signs.
