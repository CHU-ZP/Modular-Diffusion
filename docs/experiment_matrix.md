# Experiment Matrix

The repository now includes a compact experiment matrix that covers every
implemented diffusion component at least once without expanding into a full
combinatorial grid.

## Configured Experiments

| Config | Space | Backbone | Schedule | Target | Loss weighting | Sampler | Purpose |
|---|---|---|---|---|---|---|---|
| `configs/cifar10_mlp_ddpm.yaml` | Pixel | MLP | Linear | epsilon | none | DDPM | Simple pixel baseline. |
| `configs/cifar10_unet_ddpm.yaml` | Pixel | UNet | Linear | epsilon | none | DDPM | Main DDPM UNet baseline. |
| `configs/cifar10_transformer_ddpm.yaml` | Pixel | Transformer | Linear | epsilon | none | DDPM | Transformer backbone coverage. |
| `configs/cifar10_dit_ddpm.yaml` | Pixel | DiT | Linear | epsilon | none | DDPM | DiT/AdaLN-Zero backbone coverage. |
| `configs/cifar10_unet_sigmoid_ddpm.yaml` | Pixel | UNet | Sigmoid | epsilon | none | DDPM | Sigmoid schedule coverage. |
| `configs/cifar10_unet_x0_ddpm.yaml` | Pixel | UNet | Linear | x0 | none | DDPM | x0 prediction target coverage. |
| `configs/cifar10_unet_cosine.yaml` | Pixel | UNet | Cosine | epsilon | min-SNR | DDIM | Stronger pixel UNet setup. |
| `configs/cifar10_unet_snr_cosine.yaml` | Pixel | UNet | Cosine | epsilon | SNR | DDIM | SNR weighting coverage. |
| `configs/latent_unet_ddim.yaml` | Latent | UNet | Cosine | v | min-SNR | DDIM | Pretrained VAE latent diffusion. |
| `configs/latent_conv_autoencoder_smoke.yaml` | Latent | UNet | Cosine | v | min-SNR | DDIM | Built-in conv autoencoder pipeline smoke. |

## What Is Covered

- Schedules: `linear`, `cosine`, `sigmoid`
- Prediction targets: `epsilon`, `x0`, `v`
- Samplers: `DDPM`, `DDIM`
- Backbones: `MLP`, `UNet`, `Transformer`, `DiT`
- Loss weighting: none, `snr`, `min_snr`
- Representations: pixel, pretrained VAE latent, built-in conv-autoencoder latent

`latent_conv_autoencoder_smoke.yaml` is intentionally marked as a smoke config:
its autoencoder is randomly initialized and frozen, so it checks that the code
path works but is not expected to produce meaningful samples. Meaningful latent
diffusion results should use `latent_unet_ddim.yaml` with the downloaded
Diffusers VAE.
