"""Sampling entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .builders import (
    build_model,
    build_parameterization,
    build_process,
    build_representation,
    build_sampler,
    build_schedule,
    load_config,
)
from .devices import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a diffusion denoiser")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    parser.add_argument("--output", default="samples.png", help="Output image grid")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)

    schedule = build_schedule(config).to(device)
    process = build_process(schedule).to(device)
    parameterization = build_parameterization(config, schedule)
    representation = build_representation(config).to(device)
    model = build_model(config).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    sampler = build_sampler(config, process, parameterization)

    sampling_cfg = config.get("sampling", {})
    batch_size = args.batch_size or int(sampling_cfg.get("batch_size", 16))
    shape = tuple(sampling_cfg.get("shape", [batch_size, *config.get("data", {}).get("image_shape", [1, 28, 28])]))
    if shape[0] != batch_size:
        shape = (batch_size, *shape[1:])

    samples = sampler.sample(model, shape=shape, device=device)
    images = representation.decode(samples).clamp(-1.0, 1.0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, output_path, normalize=True, value_range=(-1, 1))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
