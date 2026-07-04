"""Latent diffusion representation wrapper."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentRepresentation(nn.Module):
    """Wrap an autoencoder/VAE and optional latent scaling."""

    def __init__(
        self,
        autoencoder: nn.Module,
        scaling_factor: float = 1.0,
        freeze_autoencoder: bool = True,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.scaling_factor = float(scaling_factor)
        if freeze_autoencoder:
            self.autoencoder.eval()
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad_(False)

    def encode(self, image: Tensor) -> Tensor:
        encoded = self.autoencoder.encode(image)
        if hasattr(encoded, "latent_dist"):
            latent = encoded.latent_dist.sample()
        elif hasattr(encoded, "sample") and callable(encoded.sample):
            latent = encoded.sample()
        elif isinstance(encoded, (tuple, list)):
            latent = encoded[0]
        else:
            latent = encoded
        return latent * self.scaling_factor

    def decode(self, clean_tensor: Tensor) -> Tensor:
        latent = clean_tensor / self.scaling_factor
        decoded = self.autoencoder.decode(latent)
        if hasattr(decoded, "sample"):
            sample = decoded.sample
            return sample() if callable(sample) else sample
        if isinstance(decoded, (tuple, list)):
            return decoded[0]
        if torch.is_tensor(decoded):
            return decoded
        raise TypeError("autoencoder.decode must return a tensor, tuple/list, or object with .sample")
