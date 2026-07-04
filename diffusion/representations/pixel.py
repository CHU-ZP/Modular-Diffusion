"""Pixel-space diffusion representation."""

from __future__ import annotations

from torch import Tensor, nn


class PixelRepresentation(nn.Module):
    def encode(self, image: Tensor) -> Tensor:
        return image

    def decode(self, clean_tensor: Tensor) -> Tensor:
        return clean_tensor
