"""Conditioning helpers for class-conditional diffusion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

NULL_CLASS_LABEL = -1


def apply_classifier_free_dropout(
    labels: Tensor | None,
    dropout_prob: float,
) -> Tensor | None:
    """Randomly replace class labels with the null label for CFG training."""

    if labels is None or dropout_prob <= 0.0:
        return labels
    if dropout_prob >= 1.0:
        return torch.full_like(labels.long(), NULL_CLASS_LABEL)
    labels = labels.long().clone()
    drop_mask = torch.rand(labels.shape, device=labels.device) < float(dropout_prob)
    labels[drop_mask] = NULL_CLASS_LABEL
    return labels


def class_labels_to_condition(
    labels: int | str | Sequence[int],
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """Build a batch-sized class condition tensor from a scalar or label list."""

    if isinstance(labels, str):
        values = [int(part.strip()) for part in labels.split(",") if part.strip()]
    elif isinstance(labels, int):
        values = [labels]
    else:
        values = [int(value) for value in labels]
    if not values:
        raise ValueError("at least one class label is required")

    repeats = (batch_size + len(values) - 1) // len(values)
    tiled = (values * repeats)[:batch_size]
    return torch.tensor(tiled, dtype=torch.long, device=device)
