# Latent Diffusion

Latent diffusion is represented as a separate `representation` layer. The core
diffusion math sees only tensors, so pixel diffusion and latent diffusion share
the same schedule, process, target, loss, and sampler code.

`LatentRepresentation` wraps an autoencoder with:

- `encode(image) -> latent * scaling_factor`
- `decode(latent / scaling_factor) -> image`

The autoencoder is frozen by default. For full latent experiments, download a
pretrained Diffusers `AutoencoderKL` once and load it from disk:

```bash
uv run python -m diffusion.download_vae \
  --model-id stabilityai/sd-vae-ft-mse \
  --output-dir checkpoints/vae/sd-vae-ft-mse
```

The corresponding config is:

```yaml
representation:
  type: latent
  scaling_factor: 0.18215
  freeze_autoencoder: true
  autoencoder:
    type: diffusers_autoencoder_kl
    pretrained_model_name_or_path: checkpoints/vae/sd-vae-ft-mse
    local_files_only: true
```

For CIFAR10 `32x32` images, this VAE produces `4x4` latents, so the sampler
shape should be `[batch, 4, 4, 4]`.

The repository also includes a small built-in convolutional autoencoder for
development tests:

```yaml
representation:
  type: latent
  autoencoder:
    type: conv
    image_channels: 3
    latent_channels: 4
```

If the built-in autoencoder is used without loading a trained checkpoint, it is
only useful for pipeline checks, not for meaningful latent diffusion results.
