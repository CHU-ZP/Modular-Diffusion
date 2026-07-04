"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
