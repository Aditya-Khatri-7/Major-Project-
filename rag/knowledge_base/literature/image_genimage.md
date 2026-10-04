---
modality: image
---
# GenImage: generated-image benchmark across generators

GenImage (Zhu et al., 2023) is a large collection of real images and images made by several GAN and
diffusion generators, designed to test whether a detector trained on one generator works on others,
and how it copes with resizing and compression.

The common finding is that generator-specific detectors generalise poorly to unseen generators. Face
deepfake detectors in particular are not designed for general generated scenes, so any use of
GenImage here is an optional out-of-scope stress test rather than a headline result.
