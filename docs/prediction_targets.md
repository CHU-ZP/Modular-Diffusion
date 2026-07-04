# Prediction Targets

The denoiser can be trained to predict:

- `epsilon`: the sampled noise
- `x0`: the clean input
- `v`: `sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x0`

Samplers use `DiffusionParameterization` to obtain predicted epsilon and x0
without assuming how the model was trained.
