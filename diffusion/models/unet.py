"""Small UNet denoiser baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mlp import SinusoidalTimeEmbedding


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetDenoiser(nn.Module):
    """Compact image UNet that follows denoiser(x_t, t, condition=None)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        out_channels: int | None = None,
        time_embedding_dim: int = 256,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.class_embedding = (
            nn.Embedding(num_classes, time_embedding_dim) if num_classes is not None else None
        )

        c = base_channels
        self.in_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.down_block1 = TimeResBlock(c, c, time_embedding_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.down_block2 = TimeResBlock(c * 2, c * 2, time_embedding_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)

        self.mid_block1 = TimeResBlock(c * 4, c * 4, time_embedding_dim)
        self.mid_block2 = TimeResBlock(c * 4, c * 4, time_embedding_dim)

        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up_block2 = TimeResBlock(c * 4, c * 2, time_embedding_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up_block1 = TimeResBlock(c * 2, c, time_embedding_dim)

        self.out_norm = nn.GroupNorm(_num_groups(c), c)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(self, x_t: Tensor, timesteps: Tensor, condition: Tensor | None = None) -> Tensor:
        time_embedding = self.time_mlp(timesteps)
        if self.class_embedding is not None and condition is not None:
            time_embedding = time_embedding + self.class_embedding(condition.long().view(x_t.shape[0]))

        h0 = self.in_conv(x_t)
        h1 = self.down_block1(h0, time_embedding)
        h2 = self.down_block2(self.downsample1(h1), time_embedding)
        h = self.downsample2(h2)
        h = self.mid_block2(self.mid_block1(h, time_embedding), time_embedding)

        h = self.upsample2(h)
        if h.shape[-2:] != h2.shape[-2:]:
            h = F.interpolate(h, size=h2.shape[-2:], mode="nearest")
        h = self.up_block2(torch.cat([h, h2], dim=1), time_embedding)

        h = self.upsample1(h)
        if h.shape[-2:] != h1.shape[-2:]:
            h = F.interpolate(h, size=h1.shape[-2:], mode="nearest")
        h = self.up_block1(torch.cat([h, h1], dim=1), time_embedding)
        return self.out_conv(F.silu(self.out_norm(h)))
