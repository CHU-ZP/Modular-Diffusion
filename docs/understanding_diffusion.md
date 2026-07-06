# Understanding Diffusion Through This Repository

This repository is built to demonstrate my understanding of diffusion models by
turning the main mathematical ideas into small, composable code modules. The
goal is not only to train a model on CIFAR10, but also to make each conceptual
piece visible: how noise is added, what the network learns, how samplers reverse
the process, and how pixel diffusion and latent diffusion share the same core
machinery.

My view of diffusion models is that generation is controlled by three connected
parts:

- a forward process that defines how clean data is gradually corrupted;
- a denoising network that learns what information is missing at each noise
  level;
- a sampler that decides how to use the network to move from noise back to data.

This repository mirrors that structure directly:

| Concept | Implementation |
|---|---|
| Noise schedule | `diffusion/schedules.py` |
| Forward process and posterior math | `diffusion/processes.py` |
| Prediction target conversion | `diffusion/parameterizations.py` |
| Training objective | `diffusion/losses.py` |
| Denoising networks | `diffusion/models/` |
| Reverse samplers | `diffusion/samplers.py` |
| Pixel or latent data space | `diffusion/representations/` |

## 1. Forward Noising: Defining The Problem

The forward process is fixed. It is not learned by the model. Starting from a
clean sample `x0`, diffusion repeatedly injects Gaussian noise until the data is
nearly indistinguishable from pure noise.

One step of the forward Markov chain is:

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}\!\left(x_t; \sqrt{\alpha_t}\,x_{t-1}, \beta_t I\right),
\qquad
\alpha_t = 1 - \beta_t .
$$

Equivalently:

$$
x_t =
\sqrt{\alpha_t}\,x_{t-1} +
\sqrt{\beta_t}\,\epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0, I).
$$

The noise schedule is the sequence of beta values. It determines how quickly
the signal disappears. The cumulative signal retention is:

$$
\bar{\alpha}_t =
\prod_{s=1}^{t} \alpha_s .
$$

Because all transitions are Gaussian, we do not need to simulate every previous
step during training. We can sample any noisy timestep directly:

$$
x_t =
\sqrt{\bar{\alpha}_t}\,x_0 +
\sqrt{1 - \bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
$$

I interpret this equation as a clean separation between signal and noise:

- `sqrt(alpha_bar_t) * x0` is the part of the original image that remains;
- `sqrt(1 - alpha_bar_t) * epsilon` is the noise injected at that level.

The signal-to-noise ratio makes this tradeoff explicit:

$$
\mathrm{SNR}(t) =
\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}.
$$

In the code, this is implemented by `NoiseSchedule` and
`DiffusionProcess.q_sample`. The repository supports `linear`, `cosine`, and
`sigmoid` schedules so that schedule choice is an explicit experimental
variable.

## 2. What The Network Learns

The network receives a noisy tensor `x_t` and the timestep `t`. Its job is to
predict information that helps recover the clean sample. There are several
equivalent ways to define that prediction target:

| Target | Implementation name | Meaning |
|---|---|---|
| Noise | `epsilon` | Predict the Gaussian noise added to `x0`. |
| Clean sample | `x0` | Predict the denoised tensor directly. |
| Velocity | `v` | Predict a rotated target that mixes noise and signal. |

All three targets are connected by the same forward equation:

$$
x_t = a_t x_0 + s_t \epsilon,
\qquad
a_t = \sqrt{\bar{\alpha}_t},
\qquad
s_t = \sqrt{1 - \bar{\alpha}_t}.
$$

If the model predicts noise, the clean sample can be estimated as:

$$
\hat{x}_0 =
\frac{x_t - s_t \epsilon_\theta(x_t, t)}{a_t}.
$$

If the model predicts the clean sample, the noise can be recovered as:

$$
\hat{\epsilon} =
\frac{x_t - a_t \hat{x}_0}{s_t}.
$$

For velocity prediction:

$$
v_t = a_t \epsilon - s_t x_0,
$$

and the conversions are:

$$
\hat{x}_0 = a_t x_t - s_t v_\theta,
\qquad
\hat{\epsilon} = s_t x_t + a_t v_\theta .
$$

This repository implements these conversions in
`DiffusionParameterization`. That design choice is important: the sampler should
not care whether the model was trained to predict epsilon, x0, or velocity. It
should ask for the representation it needs.

The standard training objective for epsilon prediction is:

$$
\mathcal{L} =
\mathbb{E}_{x_0,t,\epsilon}
\left\|\epsilon -
\epsilon_\theta(x_t, t)\right\|^2 .
$$

In this repository, `DiffusionLoss` generalizes this idea across prediction
targets. It also supports optional `snr` and `min_snr` weighting, because the
relative importance of low-noise and high-noise timesteps can affect stability
and final sample quality.

## 3. Samplers: Turning Denoising Into Generation

A trained network does not generate images by itself. It provides local
denoising information. The sampler decides how to use that information to move
through the noise levels.

This is the core separation I want this repository to make clear:

> The denoiser predicts what is missing; the sampler defines how to travel from
> noise to data.

The same trained denoiser can be paired with different samplers.

### DDPM

DDPM sampling treats the reverse process as a stochastic Markov chain:

$$
p_\theta(x_{t-1} \mid x_t) =
\mathcal{N}\!\left(\mu_\theta(x_t, t), \sigma_t^2 I\right).
$$

Each reverse step samples:

$$
x_{t-1} =
\mu_\theta(x_t, t) + \sigma_t z,
\qquad
z \sim \mathcal{N}(0, I).
$$

In this repository, `DDPMSampler` uses the predicted x0, computes the DDPM
posterior through `DiffusionProcess.q_posterior`, and injects stochastic noise
except at the final step. This sampler is conceptually direct and faithful to
the original discrete-time formulation, but it usually needs many steps.

### DDIM

DDIM uses the model's predicted clean sample and noise direction to move across
a chosen timestep sequence:

$$
x_\tau =
\sqrt{\bar{\alpha}_\tau}\,\hat{x}_0 +
\sqrt{1 - \bar{\alpha}_\tau}\,\hat{\epsilon}.
$$

With `eta = 0`, DDIM becomes deterministic. With fewer `num_steps`, it can
sample much faster than DDPM. In this repository, `DDIMSampler` is used for the
stronger CIFAR10 UNet config and the latent diffusion config.

### ODE-Based Samplers

ODE-style samplers view generation as continuous dynamics:

$$
\frac{dx}{dt} = f_\theta(x, t).
$$

A simple Euler-style update would be:

$$
x_{t_{i-1}} =
x_{t_i} + h_i f_\theta(x_{t_i}, t_i).
$$

Samplers such as Euler, Heun, PNDM/PLMS, DPM-Solver, and UniPC follow this
general idea. They are not implemented in this repository yet, but the current
parameterization interface is designed so that future samplers can reuse the
same denoiser outputs without assuming a single prediction target.

## 4. Pixel Diffusion And Latent Diffusion

The diffusion math only requires tensors. It does not care whether those tensors
are pixels or compressed latents. This is why the repository has a
`representation` layer.

Pixel diffusion is the simplest case:

```text
image -> PixelRepresentation.encode -> same image tensor
```

Latent diffusion adds an autoencoder:

```text
image -> VAE encoder -> latent -> diffusion process
latent -> VAE decoder -> image
```

The current latent experiment uses a pretrained Diffusers `AutoencoderKL` saved
at:

```text
checkpoints/vae/sd-vae-ft-mse
```

For CIFAR10 images of shape `3x32x32`, this VAE produces latents of shape
`4x4x4`. Therefore the latent diffusion model is trained and sampled in a much
smaller tensor space:

```yaml
sampling:
  shape: [16, 4, 4, 4]
```

This demonstrates the key idea of latent diffusion: move the expensive
generative process into a compressed representation, then decode the final
latent back to image space.

## 5. Training And Sampling In This Codebase

The training flow is:

```text
image
  -> representation.encode
  -> clean tensor
  -> sample timestep t and noise epsilon
  -> q_sample(clean, t, epsilon)
  -> x_t
  -> denoiser(x_t, t)
  -> prediction target
  -> loss
```

The sampling flow is:

```text
noise x_T
  -> DDPM or DDIM sampler
  -> clean tensor
  -> representation.decode
  -> image
```

The main experiments are CIFAR10-only, and every config trains with
classifier-free class conditioning. This means each trained checkpoint can be
sampled with the learned null condition or with CIFAR10 labels and a guidance
scale:

- `configs/cifar10_mlp_ddpm.yaml` tests a simple MLP denoiser in pixel space.
- `configs/cifar10_unet_ddpm.yaml` tests a UNet with the original DDPM-style
  setup.
- `configs/cifar10_unet_cosine.yaml` tests a stronger UNet setup with cosine
  schedule, min-SNR weighting, and DDIM sampling.
- `configs/latent_unet_ddim.yaml` tests latent diffusion with a pretrained VAE,
  velocity prediction, and DDIM sampling.

## 6. What This Repository Is Meant To Show

This project is organized to show that I understand diffusion as a composition
of separable choices:

- the noise schedule defines the corruption process;
- the prediction target defines what the network learns;
- the denoiser backbone defines model capacity;
- the sampler defines the reverse trajectory;
- the representation defines the space where diffusion happens.

Keeping these pieces separate makes the implementation easier to reason about
and makes experiments more controlled. For example, I can compare DDPM and DDIM
without changing the model, or compare epsilon and velocity prediction without
rewriting the sampler.

That modularity is the main point of the repository: each file is small, but the
boundaries match the conceptual structure of diffusion models.
