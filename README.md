# Modular Diffusion

A small diffusion research playground with stable math interfaces and replaceable
components:

- schedules: linear, cosine, sigmoid
- prediction targets: epsilon, x0, v
- samplers: DDPM, DDIM
- backbones: MLP, UNet, DiT with AdaLN-Zero blocks
- representations: pixel identity and configurable latent autoencoder wrapper
- conditioning: CIFAR10 class conditioning with classifier-free guidance

Conceptual write-up:

- [Understanding Diffusion Through This Repository](docs/understanding_diffusion.md)
- [Experiment Matrix](docs/experiment_matrix.md)
- [Classifier-Free Guidance](docs/classifier_free_guidance.md)

Run smoke tests:

```bash
uv run python -m unittest discover -s tests
```

Download the pretrained VAE used by the latent diffusion config:

```bash
uv run python -m diffusion.download_vae \
  --model-id stabilityai/sd-vae-ft-mse \
  --output-dir checkpoints/vae/sd-vae-ft-mse
```

Train a CIFAR10 baseline:

```bash
uv run python -m diffusion.train --config configs/cifar10_mlp_ddpm.yaml --device auto
```

Training checkpoints store EMA denoiser weights under `model_ema`. Sampling from
a checkpoint always loads `model_ema`.

Sample from a checkpoint unconditionally:

```bash
uv run python -m diffusion.sample --config configs/cifar10_mlp_ddpm.yaml --checkpoint runs/cifar10_mlp_ddpm/best_train_loss.pt --unconditional --output samples.uncond.png
```

Sample the same checkpoint with class guidance. The conditional grid includes
CIFAR10 label captions:

```bash
uv run python -m diffusion.sample --config configs/cifar10_mlp_ddpm.yaml --checkpoint runs/cifar10_mlp_ddpm/best_train_loss.pt --class-labels 0,1,2,3,4,5,6,7,8,9 --guidance-scale 3.0 --output samples.cond.png
```
