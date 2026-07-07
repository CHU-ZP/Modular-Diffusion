"""Exponential moving average helpers for model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel(nn.Module):
    """Maintain a non-trainable exponential moving average copy of a model."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        warmup_min_decay: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if int(warmup_steps) < 0:
            raise ValueError("EMA warmup_steps must be non-negative")
        if not 0.0 <= float(warmup_min_decay) <= float(decay):
            raise ValueError("EMA warmup_min_decay must be in [0, decay]")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.warmup_min_decay = float(warmup_min_decay)
        self.num_updates = 0
        self.module = copy.deepcopy(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @property
    def effective_decay(self) -> float:
        """Current decay after applying the linear warmup schedule."""

        if self.warmup_steps == 0 or self.num_updates >= self.warmup_steps:
            return self.decay
        progress = self.num_updates / max(1, self.warmup_steps)
        return self.warmup_min_decay + (self.decay - self.warmup_min_decay) * progress

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self.effective_decay
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach().to(
                device=ema_value.device,
                dtype=ema_value.dtype,
            )
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.module.state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - EMA is not called directly.
        return self.module(*args, **kwargs)
