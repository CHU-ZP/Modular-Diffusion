"""Conditioning modules shared by denoiser backbones."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diffusion.conditioning import NULL_CLASS_LABEL


class ClassConditionEmbedding(nn.Module):
    """Class embedding with an extra learned null token for CFG."""

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.null_index = self.num_classes
        self.embedding = nn.Embedding(self.num_classes + 1, embedding_dim)

    def forward(
        self,
        condition: Tensor | None,
        batch_size: int,
        device: torch.device | str,
    ) -> Tensor:
        if condition is None:
            indices = torch.full(
                (batch_size,),
                self.null_index,
                dtype=torch.long,
                device=device,
            )
        else:
            raw_indices = condition.long().view(batch_size).to(device)
            invalid = (raw_indices != NULL_CLASS_LABEL) & (
                (raw_indices < 0) | (raw_indices >= self.num_classes)
            )
            if torch.any(invalid):
                raise ValueError(
                    f"class labels must be in [0, {self.num_classes - 1}] "
                    f"or {NULL_CLASS_LABEL} for null conditioning",
                )
            null = torch.full_like(raw_indices, self.null_index)
            indices = torch.where(raw_indices == NULL_CLASS_LABEL, null, raw_indices)
        return self.embedding(indices)
