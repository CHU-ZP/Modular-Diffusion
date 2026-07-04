"""Forward diffusion process and posterior coefficients."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .schedules import NoiseSchedule


class DiffusionProcess(nn.Module):
    """Math for q(x_t | x_0) and q(x_{t-1} | x_t, x_0)."""

    def __init__(self, schedule: NoiseSchedule) -> None:
        super().__init__()
        self.schedule = schedule

        betas = schedule.betas
        alpha_bars = schedule.alpha_bars
        alpha_bars_prev = schedule.alpha_bars_prev
        alphas = schedule.alphas

        posterior_variance = betas * (1.0 - alpha_bars_prev) / torch.clamp(
            1.0 - alpha_bars,
            min=1e-20,
        )
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / torch.clamp(
            1.0 - alpha_bars,
            min=1e-20,
        )
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / torch.clamp(
            1.0 - alpha_bars,
            min=1e-20,
        )

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x0: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon."""

        if noise is None:
            noise = torch.randn_like(x0)
        timesteps = timesteps.to(x0.device)
        sqrt_alpha_bar = self.schedule.extract(
            self.schedule.sqrt_alpha_bars,
            timesteps,
            x0.shape,
        )
        sqrt_one_minus = self.schedule.extract(
            self.schedule.sqrt_one_minus_alpha_bars,
            timesteps,
            x0.shape,
        )
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def q_posterior(self, x0: Tensor, x_t: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return posterior mean, variance, and clipped log variance."""

        timesteps = timesteps.to(x_t.device)
        mean = (
            self.schedule.extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0
            + self.schedule.extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        variance = self.schedule.extract(self.posterior_variance, timesteps, x_t.shape)
        log_variance = self.schedule.extract(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t.shape,
        )
        return mean, variance, log_variance
