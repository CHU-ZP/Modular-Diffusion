"""Exponential moving average helpers for model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel(nn.Module):
    """Maintain a non-trainable exponential moving average copy of a model."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        super().__init__()
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach().to(
                device=ema_value.device,
                dtype=ema_value.dtype,
            )
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.module.state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - EMA is not called directly.
        return self.module(*args, **kwargs)
