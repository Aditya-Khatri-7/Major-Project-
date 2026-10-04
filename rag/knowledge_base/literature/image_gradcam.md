---
modality: image
---
# Grad-CAM heatmaps as explanations

Grad-CAM (Selvaraju et al., 2017) produces a heatmap by weighting the last convolutional feature
maps with the gradient of the class score, showing which image regions pushed the classifier
towards its decision. In this project it highlights where the EfficientNet detector found evidence
of a fake.

Caution for reports: the heatmap shows the model's attention, not a verified manipulation mask. A
hot region means the detector relied on it, which is useful for an analyst to inspect, but it is not
proof that the region was edited.
