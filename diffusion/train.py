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
from .conditioning import apply_classifier_free_dropout
from .data import build_cifar10_dataloader
from .devices import resolve_device
from .ema import EMAModel


def build_dataloader(config: dict) -> torch.utils.data.DataLoader:
    data_cfg = dict(config.get("data", {}))
    dataset = data_cfg.pop("type", "cifar10").lower()
    data_cfg.pop("image_shape", None)
    data_cfg.pop("num_classes", None)
    if dataset == "cifar10":
        return build_cifar10_dataloader(**data_cfg)
    raise ValueError(f"unknown dataset: {dataset}; this repository uses cifar10 only")


def checkpoint_payload(
    model_ema: EMAModel,
    config: dict,
    epoch: int,
    step: int,
    epoch_train_loss: float,
    best_train_loss: float,
) -> dict:
    return {
        "model_ema": model_ema.state_dict(),
        "config": config,
        "epoch": epoch,
        "step": step,
        "epoch_train_loss": float(epoch_train_loss),
        "best_train_loss": float(best_train_loss),
        "ema_decay": model_ema.decay,
    }


def save_checkpoint(
    path: Path,
    model_ema: EMAModel,
    config: dict,
    epoch: int,
    step: int,
    epoch_train_loss: float,
    best_train_loss: float,
) -> None:
    torch.save(
        checkpoint_payload(
            model_ema,
            config,
            epoch,
            step,
            epoch_train_loss,
            best_train_loss,
        ),
        path,
    )


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
    ema_cfg = training_cfg.get("ema", {})
    ema_decay = float(ema_cfg.get("decay", 0.9999))
    conditioning_cfg = config.get("conditioning", {})
    use_labels = conditioning_cfg.get("type") == "class"
    condition_dropout = float(conditioning_cfg.get("dropout_prob", 0.0))

    step = 0
    best_train_loss = float("inf")
    model_ema = EMAModel(model, decay=ema_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device) if use_labels else None
            labels = apply_classifier_free_dropout(labels, condition_dropout)
            clean = representation.encode(images)
            loss = loss_fn(model, clean, condition=labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model_ema.update(model)

            loss_value = float(loss.detach().item())
            epoch_loss_sum += loss_value
            epoch_batches += 1
            if step % log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss_value:.6f}")
            step += 1

        if epoch_batches == 0:
            raise RuntimeError("dataloader produced no batches")

        epoch_train_loss = epoch_loss_sum / epoch_batches
        is_best = epoch_train_loss < best_train_loss
        if is_best:
            best_train_loss = epoch_train_loss

        last_path = output_dir / "last.pt"
        save_checkpoint(
            last_path,
            model_ema,
            config,
            epoch,
            step,
            epoch_train_loss,
            best_train_loss,
        )
        print(f"saved {last_path} epoch_train_loss={epoch_train_loss:.6f}")

        if is_best:
            best_path = output_dir / "best_train_loss.pt"
            save_checkpoint(
                best_path,
                model_ema,
                config,
                epoch,
                step,
                epoch_train_loss,
                best_train_loss,
            )
            print(f"saved {best_path} best_train_loss={best_train_loss:.6f}")

        if epoch % save_every == 0:
            checkpoint_path = output_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                model_ema,
                config,
                epoch,
                step,
                epoch_train_loss,
                best_train_loss,
            )
            print(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
