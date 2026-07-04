"""Patch transformer denoiser backbone."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .mlp import SinusoidalTimeEmbedding


class TransformerDenoiser(nn.Module):
    """Simple ViT-style denoiser for image-shaped tensors."""

    def __init__(
        self,
        input_shape: Sequence[int],
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        time_embedding_dim: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        channels, height, width = (int(v) for v in input_shape)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("height and width must be divisible by patch_size")
        self.input_shape = (channels, height, width)
        self.patch_size = int(patch_size)
        self.grid_size = (height // patch_size, width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        time_embedding_dim = embed_dim if time_embedding_dim is None else time_embedding_dim

        self.patch_embed = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embedding = nn.Embedding(num_classes, embed_dim) if num_classes is not None else None

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.to_patch = nn.Linear(embed_dim, channels * patch_size * patch_size)

    def forward(self, x_t: Tensor, timesteps: Tensor, condition: Tensor | None = None) -> Tensor:
        batch_size = x_t.shape[0]
        tokens = self.patch_embed(x_t).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding
        context = self.time_embedding(timesteps).unsqueeze(1)
        if self.class_embedding is not None and condition is not None:
            context = context + self.class_embedding(condition.long().view(batch_size)).unsqueeze(1)
        tokens = self.encoder(tokens + context)
        patches = self.to_patch(tokens)
        return self._unpatchify(patches)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        batch_size = patches.shape[0]
        channels, _, _ = self.input_shape
        grid_h, grid_w = self.grid_size
        patch = self.patch_size
        x = patches.view(batch_size, grid_h, grid_w, channels, patch, patch)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(batch_size, channels, grid_h * patch, grid_w * patch)
