---
modality: general
---
# Calibration and why fusion needs it

A detector's raw output is often over-confident. Temperature scaling (Guo et al., 2017) divides the
logits by a single learned constant so that predicted probabilities better match observed accuracy.
Expected Calibration Error (ECE) measures the gap between confidence and accuracy across bins.

Fusing tools that each report differently scaled scores gives meaningless averages. Here each local
tool's score is calibrated on held-out data, fusion weights come from validation AUROC, and decision
thresholds are chosen on the validation precision-recall curve rather than fixed at 0.5. The reliability
diagram and ECE are reported for every tool and for the fused score.
