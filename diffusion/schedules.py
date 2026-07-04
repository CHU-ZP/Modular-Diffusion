"""Noise schedules for discrete-time diffusion."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


class NoiseSchedule(nn.Module):
    """Precomputed beta/alpha curves and scalar extraction helpers."""

    def __init__(self, betas: Tensor, name: str = "custom") -> None:
        super().__init__()
        if betas.ndim != 1:
            raise ValueError("betas must be a 1D tensor")
        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("all betas must be in the open interval (0, 1)")

        betas = betas.float()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=betas.dtype), alpha_bars[:-1]])

        self.name = name
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(torch.clamp(1.0 - alpha_bars, min=1e-20)),
        )
        self.register_buffer(
            "snr",
            alpha_bars / torch.clamp(1.0 - alpha_bars, min=1e-20),
        )

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def extract(self, values: Tensor, timesteps: Tensor, target_shape: Iterable[int]) -> Tensor:
        """Gather per-timestep scalars and reshape for broadcasting."""

        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        values = values.to(timesteps.device)
        timesteps = timesteps.long()
        timesteps = timesteps.clamp(0, values.shape[0] - 1)
        out = values.gather(0, timesteps)
        while out.ndim < len(tuple(target_shape)):
            out = out.unsqueeze(-1)
        return out


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alpha_bars = torch.cos(((x / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, min=1e-8, max=max_beta)


def sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-5,
    beta_end: float = 2e-2,
    tau: float = 1.0,
) -> Tensor:
    x = torch.linspace(-6, 6, num_timesteps)
    betas = torch.sigmoid(x / tau) * (beta_end - beta_start) + beta_start
    return torch.clamp(betas, min=1e-8, max=0.999)


def make_schedule(schedule_type: str, num_timesteps: int, **kwargs: object) -> NoiseSchedule:
    schedule_type = schedule_type.lower()
    if schedule_type == "linear":
        betas = linear_beta_schedule(num_timesteps, **kwargs)
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(num_timesteps, **kwargs)
    elif schedule_type == "sigmoid":
        betas = sigmoid_beta_schedule(num_timesteps, **kwargs)
    else:
        raise ValueError(f"unknown schedule type: {schedule_type}")
    return NoiseSchedule(betas, name=schedule_type)
