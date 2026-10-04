---
modality: text
---
# MAGE: machine-generated text detection in the wild

MAGE (Li et al., 2024, earlier released as DeepfakeTextDetect) gathers human text from many domains
and machine text from a large set of language models of different sizes and families. It defines
test settings of increasing difficulty, including unseen domains and unseen generators, and versions
with paraphrased text.

The reported pattern is that detection is comparatively easy when train and test share domain and
generator, and much harder out of distribution. This makes MAGE a good cross-domain, cross-generator
test for this project: train on HC3 only, then measure how far each tool drops on MAGE.
