# Latent Diffusion

Latent diffusion is represented as a separate `representation` layer. The core
diffusion math sees only tensors, so pixel diffusion and latent diffusion share
the same schedule, process, target, loss, and sampler code.

`LatentRepresentation` wraps an autoencoder with:

- `encode(image) -> latent * scaling_factor`
- `decode(latent / scaling_factor) -> image`

The autoencoder is frozen by default. A config can either provide an external
autoencoder object through `build_representation(config, autoencoder=...)` or
declare a built-in convolutional autoencoder:

```yaml
representation:
  type: latent
  scaling_factor: 0.18215
  freeze_autoencoder: true
  autoencoder:
    type: conv
    image_channels: 3
    latent_channels: 4
    checkpoint: path/to/autoencoder.pt
```

If `checkpoint` is omitted, the built-in autoencoder is initialized from
scratch. For meaningful latent diffusion results, load a trained autoencoder.
