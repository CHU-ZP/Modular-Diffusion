# Modular Diffusion

A small diffusion research playground with stable math interfaces and replaceable
components:

- schedules: linear, cosine, sigmoid
- prediction targets: epsilon, x0, v
- samplers: DDPM, DDIM
- backbones: MLP, UNet, Transformer, DiT with AdaLN-Zero blocks
- representations: pixel identity and configurable latent autoencoder wrapper

Conceptual write-up:

- [Understanding Diffusion Through This Repository](docs/understanding_diffusion.md)

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

Sample from a checkpoint:

```bash
uv run python -m diffusion.sample --config configs/cifar10_mlp_ddpm.yaml --checkpoint runs/cifar10_mlp_ddpm/epoch_0001.pt --output samples.png
```
