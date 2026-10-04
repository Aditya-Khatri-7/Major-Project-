---
modality: image
---
# Celeb-DF: higher-quality deepfakes

Celeb-DF (Li et al., 2020) contains celebrity face-swap videos produced with an improved synthesis
pipeline, so the fakes show fewer of the obvious visual artifacts found in earlier datasets such as
early FaceForensics++ fakes. Detectors that score highly on older datasets typically lose accuracy
on Celeb-DF.

It is therefore a harder cross-dataset test. A large drop from the training dataset to Celeb-DF is an
expected finding, not a bug, and should be reported honestly along with the in-domain result.
