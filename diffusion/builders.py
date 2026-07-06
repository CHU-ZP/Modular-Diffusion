"""YAML-driven component builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .losses import DiffusionLoss
from .models import (
    ConvAutoencoder,
    DiTDenoiser,
    MLPDenoiser,
    TransformerDenoiser,
    UNetDenoiser,
    load_diffusers_autoencoder_kl,
)
from .parameterizations import DiffusionParameterization
from .processes import DiffusionProcess
from .representations import LatentRepresentation, PixelRepresentation
from .samplers import DDIMSampler, DDPMSampler
from .schedules import NoiseSchedule, make_schedule


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return {} if value is None else dict(value)


def build_schedule(config: dict[str, Any]) -> NoiseSchedule:
    cfg = _section(config, "schedule") if "schedule" in config else dict(config)
    cfg = dict(cfg)
    schedule_type = cfg.pop("type", "linear")
    num_timesteps = int(cfg.pop("num_timesteps", 1000))
    return make_schedule(schedule_type, num_timesteps=num_timesteps, **cfg)


def build_process(schedule: NoiseSchedule) -> DiffusionProcess:
    return DiffusionProcess(schedule)


def build_parameterization(
    config: dict[str, Any],
    schedule: NoiseSchedule,
) -> DiffusionParameterization:
    cfg = _section(config, "diffusion") if "diffusion" in config else dict(config)
    return DiffusionParameterization(
        schedule,
        prediction_target=cfg.get("prediction_target", "epsilon"),
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    cfg = _section(config, "model") if "model" in config else dict(config)
    data_cfg = _section(config, "data")
    cfg = dict(cfg)
    model_type = cfg.pop("type", "mlp").lower()
    input_shape = cfg.pop("input_shape", data_cfg.get("image_shape", [1, 28, 28]))
    num_classes = cfg.pop("num_classes", data_cfg.get("num_classes"))

    if model_type == "mlp":
        return MLPDenoiser(input_shape=input_shape, num_classes=num_classes, **cfg)
    if model_type == "unet":
        in_channels = int(cfg.pop("in_channels", input_shape[0]))
        return UNetDenoiser(in_channels=in_channels, num_classes=num_classes, **cfg)
    if model_type == "transformer":
        return TransformerDenoiser(input_shape=input_shape, num_classes=num_classes, **cfg)
    if model_type == "dit":
        return DiTDenoiser(input_shape=input_shape, num_classes=num_classes, **cfg)
    raise ValueError(f"unknown model type: {model_type}")


def _load_module_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    state_dict_key: str | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if state_dict_key is not None:
        checkpoint = checkpoint[state_dict_key]
    elif isinstance(checkpoint, dict):
        if "autoencoder" in checkpoint:
            checkpoint = checkpoint["autoencoder"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state dict")
    module.load_state_dict(checkpoint)


def build_autoencoder(config: dict[str, Any]) -> nn.Module:
    representation_cfg = _section(config, "representation")
    data_cfg = _section(config, "data")
    model_cfg = _section(config, "model")
    sampling_cfg = _section(config, "sampling")
    cfg = dict(representation_cfg.get("autoencoder") or {})
    autoencoder_type = cfg.pop("type", "conv").lower()
    checkpoint_path = cfg.pop("checkpoint", None)
    state_dict_key = cfg.pop("state_dict_key", None)

    if autoencoder_type in {"diffusers_autoencoder_kl", "autoencoder_kl"}:
        pretrained_model_name_or_path = cfg.pop(
            "pretrained_model_name_or_path",
            checkpoint_path,
        )
        if pretrained_model_name_or_path is None:
            raise ValueError(
                "diffusers_autoencoder_kl requires pretrained_model_name_or_path",
            )
        return load_diffusers_autoencoder_kl(
            pretrained_model_name_or_path,
            subfolder=cfg.pop("subfolder", None),
            revision=cfg.pop("revision", None),
            variant=cfg.pop("variant", None),
            torch_dtype=cfg.pop("torch_dtype", None),
            local_files_only=bool(cfg.pop("local_files_only", False)),
            cache_dir=cfg.pop("cache_dir", None),
        )

    if autoencoder_type != "conv":
        raise ValueError(f"unknown autoencoder type: {autoencoder_type}")

    image_shape = data_cfg.get("image_shape", [3, 32, 32])
    sampling_shape = sampling_cfg.get("shape", [])
    cfg.setdefault("image_channels", int(image_shape[0]))
    cfg.setdefault(
        "latent_channels",
        int(model_cfg.get("in_channels", sampling_shape[1] if len(sampling_shape) > 1 else 4)),
    )
    autoencoder = ConvAutoencoder(**cfg)
    if checkpoint_path is not None:
        _load_module_checkpoint(autoencoder, checkpoint_path, state_dict_key=state_dict_key)
    return autoencoder


def build_representation(
    config: dict[str, Any],
    autoencoder: nn.Module | None = None,
) -> nn.Module:
    cfg = _section(config, "representation") if "representation" in config else dict(config)
    representation_type = cfg.get("type", "pixel").lower()
    if representation_type == "pixel":
        return PixelRepresentation()
    if representation_type == "latent":
        if autoencoder is None:
            autoencoder = build_autoencoder(config)
        return LatentRepresentation(
            autoencoder,
            scaling_factor=cfg.get("scaling_factor", 1.0),
            freeze_autoencoder=cfg.get("freeze_autoencoder", True),
        )
    raise ValueError(f"unknown representation type: {representation_type}")


def build_loss(
    config: dict[str, Any],
    process: DiffusionProcess,
    parameterization: DiffusionParameterization,
) -> DiffusionLoss:
    cfg = _section(config, "loss") if "loss" in config else dict(config)
    return DiffusionLoss(
        process,
        parameterization,
        loss_type=cfg.get("type", "mse"),
        weighting=cfg.get("weighting"),
        min_snr_gamma=cfg.get("min_snr_gamma", 5.0),
    )


def build_sampler(
    config: dict[str, Any],
    process: DiffusionProcess,
    parameterization: DiffusionParameterization,
) -> DDPMSampler | DDIMSampler:
    cfg = _section(config, "sampler") if "sampler" in config else dict(config)
    cfg = dict(cfg)
    sampler_type = cfg.pop("type", "ddpm").lower()
    if sampler_type == "ddpm":
        return DDPMSampler(process, parameterization, **cfg)
    if sampler_type == "ddim":
        return DDIMSampler(process, parameterization, **cfg)
    raise ValueError(f"unknown sampler type: {sampler_type}")


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    cfg = _section(config, "training")
    optimizer_cfg = dict(cfg.get("optimizer", {"type": "adamw", "lr": 1e-4}))
    optimizer_type = optimizer_cfg.pop("type", "adamw").lower()
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_cfg)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_cfg)
    raise ValueError(f"unknown optimizer type: {optimizer_type}")
