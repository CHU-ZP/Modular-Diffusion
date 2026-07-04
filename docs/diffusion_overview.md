# Diffusion Overview

The repository is split around stable mathematical boundaries:

- `schedules.py` owns beta, alpha, alpha_bar, and SNR curves.
- `processes.py` owns the forward process and DDPM posterior coefficients.
- `parameterizations.py` owns epsilon, x0, and v target conversion.
- `models/` owns denoiser backbones only.
- `losses.py` owns training supervision and optional weighting.
- `samplers.py` owns reverse-time generation.
- `representations/` owns the space where diffusion happens.

Training data flow:

```text
image -> representation.encode -> clean -> q_sample -> x_t
      -> denoiser(x_t, t, condition) -> target loss
```

Sampling data flow:

```text
noise x_T -> sampler loop -> clean tensor -> representation.decode -> image
```
