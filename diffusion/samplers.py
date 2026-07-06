"""Reverse diffusion samplers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .parameterizations import DiffusionParameterization
from .processes import DiffusionProcess


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback


def _guided_model_output(
    denoiser: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    condition: Tensor | None = None,
    guidance_scale: float = 1.0,
) -> Tensor:
    if condition is None:
        return denoiser(x_t, timesteps, None)
    condition = condition.to(x_t.device)
    if float(guidance_scale) == 1.0:
        return denoiser(x_t, timesteps, condition)

    unconditional = denoiser(x_t, timesteps, None)
    conditional = denoiser(x_t, timesteps, condition)
    return unconditional + float(guidance_scale) * (conditional - unconditional)


class DDPMSampler:
    """Original stochastic DDPM reverse sampler."""

    def __init__(
        self,
        process: DiffusionProcess,
        parameterization: DiffusionParameterization,
        clip_x0: bool = True,
    ) -> None:
        self.process = process
        self.parameterization = parameterization
        self.clip_x0 = clip_x0

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        condition: Tensor | None = None,
        guidance_scale: float = 1.0,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        device = torch.device(device) if device is not None else _module_device(
            denoiser,
            self.process.schedule.betas.device,
        )
        x = torch.randn(shape, device=device, generator=generator)
        trajectory: list[Tensor] = []

        for step in reversed(range(self.process.schedule.num_timesteps)):
            timesteps = torch.full((shape[0],), step, dtype=torch.long, device=device)
            model_output = _guided_model_output(
                denoiser,
                x,
                timesteps,
                condition,
                guidance_scale,
            )
            predicted_x0 = self.parameterization.model_output_to_x0(model_output, x, timesteps)
            if self.clip_x0:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
            mean, _, log_variance = self.process.q_posterior(predicted_x0, x, timesteps)
            if step > 0:
                noise = torch.randn(shape, device=device, generator=generator)
            else:
                noise = torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_variance) * noise
            if return_trajectory:
                trajectory.append(x.detach().cpu())

        if return_trajectory:
            return x, trajectory
        return x


class DDIMSampler:
    """DDIM sampler with optional stochasticity through eta."""

    def __init__(
        self,
        process: DiffusionProcess,
        parameterization: DiffusionParameterization,
        num_steps: int | None = None,
        eta: float = 0.0,
        clip_x0: bool = True,
    ) -> None:
        self.process = process
        self.parameterization = parameterization
        self.num_steps = num_steps
        self.eta = float(eta)
        self.clip_x0 = clip_x0

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        condition: Tensor | None = None,
        guidance_scale: float = 1.0,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        device = torch.device(device) if device is not None else _module_device(
            denoiser,
            self.process.schedule.betas.device,
        )
        x = torch.randn(shape, device=device, generator=generator)
        trajectory: list[Tensor] = []
        timesteps = self._timesteps(device)
        alpha_bars = self.process.schedule.alpha_bars.to(device)

        for index, step_tensor in enumerate(timesteps):
            step = int(step_tensor.item())
            next_step = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            batch_timesteps = torch.full((shape[0],), step, dtype=torch.long, device=device)

            model_output = _guided_model_output(
                denoiser,
                x,
                batch_timesteps,
                condition,
                guidance_scale,
            )
            predicted_epsilon = self.parameterization.model_output_to_epsilon(
                model_output,
                x,
                batch_timesteps,
            )
            predicted_x0 = self.parameterization.model_output_to_x0(model_output, x, batch_timesteps)
            if self.clip_x0:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

            alpha_t = alpha_bars[step]
            alpha_prev = torch.ones((), device=device) if next_step < 0 else alpha_bars[next_step]
            variance = (
                (1.0 - alpha_prev)
                / torch.clamp(1.0 - alpha_t, min=1e-20)
                * (1.0 - alpha_t / torch.clamp(alpha_prev, min=1e-20))
            ).clamp_min(0.0)
            sigma = self.eta * torch.sqrt(variance)
            noise = torch.randn(shape, device=device, generator=generator) if next_step >= 0 else 0.0
            direction = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0))
            x = torch.sqrt(alpha_prev) * predicted_x0 + direction * predicted_epsilon + sigma * noise
            if return_trajectory:
                trajectory.append(x.detach().cpu())

        if return_trajectory:
            return x, trajectory
        return x

    def _timesteps(self, device: torch.device) -> Tensor:
        total = self.process.schedule.num_timesteps
        if self.num_steps is None or self.num_steps >= total:
            return torch.arange(total - 1, -1, -1, device=device, dtype=torch.long)
        return torch.linspace(total - 1, 0, self.num_steps, device=device).round().long()


def make_sampler(
    sampler_type: str,
    process: DiffusionProcess,
    parameterization: DiffusionParameterization,
    **kwargs: object,
) -> DDPMSampler | DDIMSampler:
    sampler_type = sampler_type.lower()
    if sampler_type == "ddpm":
        return DDPMSampler(process, parameterization, **kwargs)
    if sampler_type == "ddim":
        return DDIMSampler(process, parameterization, **kwargs)
    raise ValueError(f"unknown sampler type: {sampler_type}")
