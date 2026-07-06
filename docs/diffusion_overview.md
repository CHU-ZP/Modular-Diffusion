# Diffusion Overview

This repository implements a modular discrete-time diffusion pipeline. The
generation process is organized around three cooperating parts:

- a fixed forward noising process and noise schedule;
- a denoising network trained at different noise levels;
- a sampler that uses the network to turn pure noise back into data.

The code is split along these boundaries:

- `schedules.py` owns beta, alpha, alpha-bar, and SNR curves.
- `processes.py` owns `q_sample` and DDPM posterior coefficients.
- `parameterizations.py` owns epsilon, x0, and v target conversion.
- `models/` owns denoiser backbones: MLP, UNet, Transformer, and DiT.
- `losses.py` owns training supervision and optional SNR weighting.
- `samplers.py` owns reverse-time generation with DDPM and DDIM.
- `representations/` owns the space where diffusion happens: pixels or latents.

## Forward Noising And Schedules

The forward process is a fixed Markov chain. At each step:

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}\!\left(x_t; \sqrt{\alpha_t}\,x_{t-1}, \beta_t I\right),
\qquad \alpha_t = 1 - \beta_t .
$$

Equivalently:

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0, I).
$$

A noise schedule is defined by a sequence of beta values. The cumulative signal
retention is:

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s .
$$

Because the Gaussian transitions compose in closed form, training can sample any
noisy timestep directly from the clean data:

$$
x_t =
\sqrt{\bar{\alpha}_t}\,x_0 +
\sqrt{1 - \bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I).
$$

In the implementation, `NoiseSchedule` precomputes these values, and
`DiffusionProcess.q_sample` applies this direct formula. The two terms have a
simple interpretation:

- `sqrt_alpha_bar * x0` is the remaining signal.
- `sqrt_one_minus_alpha_bar * noise` is the injected noise.

The signal-to-noise ratio is:

$$
\mathrm{SNR}(t) =
\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}.
$$

This repository currently implements `linear`, `cosine`, and `sigmoid`
schedules. These are configured through YAML and built by `build_schedule`.

## Prediction Targets

Given a noisy sample `x_t` and timestep `t`, the denoiser can be trained to
predict one of three equivalent targets:

| Target | Symbol | Implementation name | Notes |
|---|---|---|---|
| Noise | $\epsilon_\theta$ | `epsilon` | Classic DDPM target. |
| Clean sample | $\hat{x}_0$ | `x0` | Predicts the denoised tensor directly. |
| Velocity | $v_\theta$ | `v` | Stable across noise scales and useful for latent diffusion. |

All three are tied together by:

$$
x_t = a_t x_0 + s_t \epsilon,
\qquad
a_t = \sqrt{\bar{\alpha}_t},
\qquad
s_t = \sqrt{1 - \bar{\alpha}_t}.
$$

The repository centralizes target conversion in `DiffusionParameterization`.
Samplers do not need to know how the model was trained; they ask the
parameterization object for predicted epsilon or predicted x0.

The key conversions are:

$$
\hat{x}_0 =
\frac{x_t - s_t \epsilon_\theta}{a_t},
\qquad
\hat{\epsilon} =
\frac{x_t - a_t \hat{x}_0}{s_t},
$$

and for velocity prediction:

$$
\hat{x}_0 = a_t x_t - s_t v_\theta,
\qquad
\hat{\epsilon} = s_t x_t + a_t v_\theta .
$$

The training loss is implemented by `DiffusionLoss`. For epsilon prediction, the
standard objective is:

$$
\mathcal{L} =
\mathbb{E}_{x_0,t,\epsilon}
\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|^2 .
$$

The code supports MSE and L1 losses, plus optional `snr` and `min_snr` weighting.

## Samplers

A sampler answers the reverse-time question: after training the denoiser, how do
we start from `x_T ~ N(0, I)` and reconstruct a data sample?

This repository currently implements two samplers.

### DDPM

`DDPMSampler` uses the ancestral DDPM reverse process:

$$
p_\theta(x_{t-1} \mid x_t) =
\mathcal{N}\!\left(\mu_\theta(x_t, t), \sigma_t^2 I\right).
$$

Each step injects fresh Gaussian noise:

$$
x_{t-1} =
\mu_\theta(x_t, t) + \sigma_t z,
\qquad z \sim \mathcal{N}(0, I).
$$

In code, the sampler obtains `predicted_x0` from the parameterization adapter,
uses `DiffusionProcess.q_posterior` to compute the posterior mean and variance,
and then samples the previous timestep. This is stochastic and usually uses all
training timesteps.

### DDIM

`DDIMSampler` uses the model's predicted x0 and epsilon to move across a
timestep subsequence:

$$
x_\tau =
\sqrt{\bar{\alpha}_\tau}\,\hat{x}_0 +
\sqrt{1 - \bar{\alpha}_\tau}\,\hat{\epsilon}.
$$

The implementation supports:

- `num_steps`: use fewer sampling steps than training timesteps;
- `eta`: control stochasticity;
- `eta = 0`: deterministic sampling.

This makes DDIM useful for faster CIFAR10 experiments and for the latent
diffusion config.

### ODE-Based Samplers

ODE-based samplers are not implemented yet in this repository. Conceptually,
they view reverse generation as continuous dynamics:

$$
\frac{dx}{dt} = f_\theta(x, t),
$$

and apply numerical integration, for example Euler:

$$
x_{t_{i-1}} =
x_{t_i} + h_i f_\theta(x_{t_i}, t_i).
$$

Euler, Heun, PNDM/PLMS, DPM-Solver, and UniPC can be added later by reusing the
same `DiffusionParameterization` interface instead of assuming epsilon
prediction.

> [!NOTE]
> The network provides denoising information, such as epsilon, x0, or velocity.
> The sampler decides how that information is converted into a trajectory through
> noise levels. The same trained network can be paired with different samplers,
> changing speed, stochasticity, and sample quality.

## Training And Sampling Flow

Training data flow:

```text
image
  -> representation.encode
  -> clean tensor
  -> q_sample(clean, t, noise)
  -> x_t
  -> denoiser(x_t, t, condition)
  -> target loss
```

Sampling data flow:

```text
noise x_T
  -> sampler loop
  -> clean tensor
  -> representation.decode
  -> image
```

The repository supports two representation spaces:

- Pixel diffusion, where `representation.encode` and `decode` are identities.
- Latent diffusion, where images are encoded and decoded by an autoencoder.

The current latent experiment uses a pretrained Diffusers `AutoencoderKL` saved
at `checkpoints/vae/sd-vae-ft-mse`. For CIFAR10 images of shape `3x32x32`, that
VAE produces latents of shape `4x4x4`, so the latent diffusion sampler uses
`shape: [batch, 4, 4, 4]`.

## CIFAR10 Experiments In This Repository

The repository is CIFAR10-only at the data layer. Current experiment configs
cover:

- `configs/cifar10_mlp_ddpm.yaml`: pixel-space MLP baseline with linear schedule
  and DDPM sampling.
- `configs/cifar10_unet_ddpm.yaml`: pixel-space UNet baseline with linear
  schedule and DDPM sampling.
- `configs/cifar10_unet_cosine.yaml`: pixel-space UNet with cosine schedule,
  min-SNR weighting, and DDIM sampling.
- `configs/latent_unet_ddim.yaml`: latent-space UNet with a pretrained VAE,
  velocity prediction, cosine schedule, min-SNR weighting, and DDIM sampling.
