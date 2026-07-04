"""Training losses for diffusion denoisers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .parameterizations import DiffusionParameterization
from .processes import DiffusionProcess


class DiffusionLoss(nn.Module):
    """Compose q_sample, target conversion, denoiser forward, and weighting."""

    def __init__(
        self,
        process: DiffusionProcess,
        parameterization: DiffusionParameterization,
        loss_type: str = "mse",
        weighting: str | None = None,
        min_snr_gamma: float = 5.0,
    ) -> None:
        super().__init__()
        self.process = process
        self.parameterization = parameterization
        self.loss_type = loss_type.lower()
        self.weighting = None if weighting in (None, "none") else weighting.lower()
        self.min_snr_gamma = float(min_snr_gamma)
        if self.loss_type not in {"mse", "l1"}:
            raise ValueError("loss_type must be 'mse' or 'l1'")
        if self.weighting not in {None, "snr", "min_snr"}:
            raise ValueError("weighting must be one of None, 'snr', or 'min_snr'")

    def forward(
        self,
        denoiser: nn.Module,
        x0: Tensor,
        condition: Tensor | None = None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        batch_size = x0.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.process.schedule.num_timesteps,
                (batch_size,),
                device=x0.device,
            )
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.process.q_sample(x0, timesteps, noise)
        model_output = denoiser(x_t, timesteps, condition)
        target = self.parameterization.target_from(x0, noise, x_t, timesteps)

        if model_output.shape != target.shape:
            raise ValueError(
                f"model output shape {tuple(model_output.shape)} does not match "
                f"target shape {tuple(target.shape)}",
            )

        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction="none")
        else:
            loss = F.l1_loss(model_output, target, reduction="none")

        loss = loss.flatten(1).mean(dim=1)
        weights = self._weights(timesteps, loss.shape)
        if weights is not None:
            loss = loss * weights.to(loss.device)
        loss_value = loss.mean()

        if return_details:
            return loss_value, {
                "timesteps": timesteps,
                "noise": noise,
                "x_t": x_t,
                "target": target,
                "model_output": model_output,
            }
        return loss_value

    def _weights(self, timesteps: Tensor, shape: torch.Size) -> Tensor | None:
        if self.weighting is None:
            return None
        snr = self.process.schedule.extract(self.process.schedule.snr, timesteps, shape).flatten()
        if self.weighting == "snr":
            return snr
        clipped = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma))
        return clipped / snr.clamp_min(1e-20)
