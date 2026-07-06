# Experiment Matrix

The repository includes a compact experiment matrix that covers the main
diffusion components without expanding into a full combinatorial grid. Formal
experiments use the full CIFAR10 training split for 100 epochs.

## Configured Experiments

| Config | Space | Backbone | Schedule | Target | Loss weighting | Sampler | Conditioning | Purpose |
|---|---|---|---|---|---|---|---|---|
| `configs/cifar10_mlp_ddpm.yaml` | Pixel | MLP | Linear | epsilon | none | DDPM | CFG class | Simple pixel baseline. |
| `configs/cifar10_unet_ddpm.yaml` | Pixel | UNet | Linear | epsilon | none | DDPM | CFG class | Main DDPM UNet baseline. |
| `configs/cifar10_transformer_ddpm.yaml` | Pixel | Transformer | Linear | epsilon | none | DDPM | CFG class | Transformer backbone coverage. |
| `configs/cifar10_dit_ddpm.yaml` | Pixel | DiT | Linear | epsilon | none | DDPM | CFG class | DiT/AdaLN-Zero backbone coverage. |
| `configs/cifar10_unet_sigmoid_ddpm.yaml` | Pixel | UNet | Sigmoid | epsilon | none | DDPM | CFG class | Sigmoid schedule coverage. |
| `configs/cifar10_unet_x0_ddpm.yaml` | Pixel | UNet | Linear | x0 | none | DDPM | CFG class | x0 prediction target coverage. |
| `configs/cifar10_unet_cosine.yaml` | Pixel | UNet | Cosine | epsilon | min-SNR | DDIM | CFG class | Stronger pixel UNet setup. |
| `configs/cifar10_unet_snr_cosine.yaml` | Pixel | UNet | Cosine | epsilon | SNR | DDIM | CFG class | SNR weighting coverage. |
| `configs/latent_unet_ddim.yaml` | Latent | UNet | Cosine | v | min-SNR | DDIM | CFG class | Pretrained VAE latent diffusion. |

## What Is Covered

- Schedules: `linear`, `cosine`, `sigmoid`
- Prediction targets: `epsilon`, `x0`, `v`
- Samplers: `DDPM`, `DDIM`
- Backbones: `MLP`, `UNet`, `Transformer`, `DiT`
- Loss weighting: none, `snr`, `min_snr`
- Representations: pixel, pretrained VAE latent
- Conditioning: every experiment trains with classifier-free class conditioning

## Smoke-Only Config

`configs/latent_conv_autoencoder_smoke.yaml` is intentionally kept as a smoke
config:
its autoencoder is randomly initialized and frozen, so it checks that the code
path works but is not expected to produce meaningful samples. It is not part of
the full experiment runner. Meaningful latent diffusion results should use
`latent_unet_ddim.yaml` with the downloaded Diffusers VAE.

The full experiment runner saves two sample grids for every checkpoint:
unconditional samples with the learned null label and guided class-conditional
samples with CIFAR10 labels.
