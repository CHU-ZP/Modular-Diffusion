"""DiT-style patch transformer denoiser."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .mlp import SinusoidalTimeEmbedding


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(context).chunk(6, dim=1)
        )

        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, embed_dim: int, patch_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.linear = nn.Linear(embed_dim, patch_dim)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(context).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class DiTDenoiser(nn.Module):
    """Compact DiT denoiser with AdaLN-Zero blocks."""

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
        patch_size = int(patch_size)
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("height and width must be divisible by patch_size")

        self.input_shape = (channels, height, width)
        self.patch_size = patch_size
        self.grid_size = (height // patch_size, width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        patch_dim = channels * patch_size * patch_size
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
        self.class_embedding = (
            nn.Embedding(num_classes, embed_dim) if num_classes is not None else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)],
        )
        self.final_layer = DiTFinalLayer(embed_dim, patch_dim)

    def forward(self, x_t: Tensor, timesteps: Tensor, condition: Tensor | None = None) -> Tensor:
        batch_size = x_t.shape[0]
        tokens = self.patch_embed(x_t).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding

        context = self.time_embedding(timesteps)
        if self.class_embedding is not None and condition is not None:
            context = context + self.class_embedding(condition.long().view(batch_size))

        for block in self.blocks:
            tokens = block(tokens, context)

        patches = self.final_layer(tokens, context)
        return self._unpatchify(patches)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        batch_size = patches.shape[0]
        channels, _, _ = self.input_shape
        grid_h, grid_w = self.grid_size
        patch = self.patch_size
        x = patches.view(batch_size, grid_h, grid_w, channels, patch, patch)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(batch_size, channels, grid_h * patch, grid_w * patch)
