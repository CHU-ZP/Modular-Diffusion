"""Adapters for autoencoders from Hugging Face Diffusers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def parse_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"auto", "none"}:
        return None
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unknown torch dtype: {dtype}")


def load_diffusers_autoencoder_kl(
    pretrained_model_name_or_path: str | Path,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    variant: str | None = None,
    torch_dtype: str | torch.dtype | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> nn.Module:
    """Load a Diffusers AutoencoderKL while keeping diffusers optional."""

    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise ImportError(
            "Diffusers AutoencoderKL support requires the optional dependencies "
            "`diffusers` and `safetensors`. Install them with `uv sync` after "
            "updating this project, or run `uv add diffusers safetensors`.",
        ) from exc

    kwargs: dict[str, object] = {
        "torch_dtype": parse_torch_dtype(torch_dtype),
        "local_files_only": local_files_only,
    }
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    if revision is not None:
        kwargs["revision"] = revision
    if variant is not None:
        kwargs["variant"] = variant
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    return AutoencoderKL.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
