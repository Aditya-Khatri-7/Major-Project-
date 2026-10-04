---
modality: image
---
# Generator fingerprint: diffusion models such as Stable Diffusion

Diffusion images tend to have overly smooth or waxy skin and surfaces, plausible but incorrect hands
and fingers, garbled or meaningless text on signs and labels, inconsistent reflections and shadows,
objects that blend into each other, and a polished, high-contrast "stock" look. Fine sensor-like noise
is usually absent or too uniform.

They show fewer of the crude high-frequency artifacts of older GANs, so artifact-based CNNs trained on
GAN data often fail on them. Semantic and physical implausibility, which a vision-language judge can
notice, is often the more informative signal, but very good generations can pass both checks.
