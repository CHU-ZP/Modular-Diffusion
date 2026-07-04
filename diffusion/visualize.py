"""Visualization helpers for samples and schedules."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image

from .schedules import NoiseSchedule


def sample_grid(samples: Tensor, nrow: int = 8) -> Tensor:
    return make_grid(samples.clamp(-1.0, 1.0), nrow=nrow, normalize=True, value_range=(-1, 1))


def save_sample_grid(samples: Tensor, path: str | Path, nrow: int = 8) -> None:
    save_image(samples.clamp(-1.0, 1.0), path, nrow=nrow, normalize=True, value_range=(-1, 1))


def schedule_curves(schedule: NoiseSchedule) -> dict[str, Tensor]:
    return {
        "beta": schedule.betas.detach().cpu(),
        "alpha_bar": schedule.alpha_bars.detach().cpu(),
        "snr": schedule.snr.detach().cpu(),
    }
