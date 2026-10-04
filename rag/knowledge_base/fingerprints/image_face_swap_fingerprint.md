---
modality: image
---
# Generator fingerprint: face-swap and reenactment (Deepfakes, FaceSwap, Face2Face, NeuralTextures)

Face-swap methods paste a synthesised face into a target frame, so evidence concentrates around the
face region: a visible or blurred blending boundary along the jaw and hairline, skin tone or lighting
mismatch between face and neck, softer face texture than the surrounding image, inconsistent eye
reflections or gaze, and irregular teeth. Reenactment methods (Face2Face, NeuralTextures) alter
expression and mouth motion and can leave texture flicker around the mouth in video.

Heavy compression hides many of these traces. A Grad-CAM heatmap concentrated on the face boundary or
mouth area is consistent with this family, though it shows model attention rather than proof.
