"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .builders import (
    build_loss,
    build_model,
    build_optimizer,
    build_parameterization,
    build_process,
    build_representation,
    build_schedule,
    load_config,
)
from .data import build_cifar10_dataloader
from .devices import resolve_device


def build_dataloader(config: dict) -> torch.utils.data.DataLoader:
    data_cfg = dict(config.get("data", {}))
    dataset = data_cfg.pop("type", "cifar10").lower()
    if dataset == "cifar10":
        return build_cifar10_dataloader(**data_cfg)
    raise ValueError(f"unknown dataset: {dataset}; this repository uses cifar10 only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a diffusion denoiser")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    parser.add_argument("--output-dir", default="runs", help="Directory for checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir) / config.get("experiment_name", "diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(config).to(device)
    process = build_process(schedule).to(device)
    parameterization = build_parameterization(config, schedule)
    representation = build_representation(config).to(device)
    model = build_model(config).to(device)
    loss_fn = build_loss(config, process, parameterization)
    optimizer = build_optimizer(config, model)
    dataloader = build_dataloader(config)

    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 1))
    log_every = int(training_cfg.get("log_every", 100))
    save_every = int(training_cfg.get("save_every", 1))
    use_labels = config.get("conditioning", {}).get("type") == "class"

    step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device) if use_labels else None
            clean = representation.encode(images)
            loss = loss_fn(model, clean, condition=labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")
            step += 1

        if epoch % save_every == 0:
            checkpoint_path = output_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "step": step,
                },
                checkpoint_path,
            )
            print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
