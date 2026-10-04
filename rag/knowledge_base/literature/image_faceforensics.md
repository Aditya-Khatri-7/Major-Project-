---
modality: image
---
# FaceForensics++ and manipulation types

FaceForensics++ (Roessler et al., 2019, arXiv:1901.08971) provides real videos and fakes made with
four manipulation methods: Deepfakes and FaceSwap (identity replacement), and Face2Face and
NeuralTextures (expression or texture reenactment). Each is available at several compression levels
(raw, c23, c40).

Key lessons: detectors are easiest on raw video and degrade markedly as compression rises, and a
detector trained on one manipulation method often transfers poorly to others. This project uses the
c23 version as a cross-dataset test and reports accuracy per manipulation method.
