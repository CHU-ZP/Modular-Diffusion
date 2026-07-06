"""MLP denoiser for image and toy tensors."""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Sequence

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1),
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class MLPDenoiser(nn.Module):
    """Flattened-image denoiser with a unified denoiser(x_t, t, condition) API."""

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_dims: Sequence[int] = (512, 512),
        time_embedding_dim: int = 128,
        condition_dim: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.flat_dim = reduce(mul, self.input_shape, 1)
        self.condition_dim = condition_dim
        self.num_classes = num_classes

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, time_embedding_dim)
            context_dim = time_embedding_dim
        else:
            self.class_embedding = None
            context_dim = int(condition_dim or 0)

        dims = [self.flat_dim + time_embedding_dim + context_dim, *hidden_dims, self.flat_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: Tensor, timesteps: Tensor, condition: Tensor | None = None) -> Tensor:
        batch_size = x_t.shape[0]
        x_flat = x_t.reshape(batch_size, -1)
        parts = [x_flat, self.time_embedding(timesteps)]

        if self.class_embedding is not None:
            if condition is None:
                parts.append(torch.zeros_like(parts[-1]))
            else:
                parts.append(self.class_embedding(condition.long().view(batch_size)))
        elif self.condition_dim:
            if condition is None:
                parts.append(torch.zeros(batch_size, self.condition_dim, device=x_t.device))
            else:
                parts.append(condition.reshape(batch_size, -1).float())

        out = self.net(torch.cat(parts, dim=1))
        return out.reshape(batch_size, *self.input_shape)
