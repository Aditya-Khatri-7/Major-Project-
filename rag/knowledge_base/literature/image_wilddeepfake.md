---
modality: image
---
# WildDeepfake: deepfakes collected from the internet

WildDeepfake (Zi et al., 2020) gathers real and fake face sequences from internet sources instead of
generating them in a lab. The fakes come from many unknown tools, with varied resolution,
compression, lighting and post-processing, which makes it more realistic and usually harder than
lab-built datasets.

It is the training and in-domain evaluation set for the image classifier here. Because it is
internet-sourced, models trained on it can partly learn source or compression cues, so cross-dataset
testing on FaceForensics++ and Celeb-DF is essential.
