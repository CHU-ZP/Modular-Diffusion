"""Sampling entry point."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
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
from .conditioning import class_labels_to_condition
from .devices import resolve_device

CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
GRID_NROW = 8


def _sampling_class_labels(args: argparse.Namespace, config: dict, batch_size: int, device: torch.device) -> torch.Tensor | None:
    class_label = getattr(args, "class_label", None)
    class_labels = getattr(args, "class_labels", None)
    if getattr(args, "unconditional", False):
        if class_label is not None or class_labels is not None:
            raise ValueError("--unconditional cannot be combined with class labels")
        return None

    if config.get("conditioning", {}).get("type") != "class":
        if class_label is not None or class_labels is not None:
            raise ValueError("class labels require `conditioning.type: class` in the config")
        return None

    sampling_cfg = config.get("sampling", {})
    labels = class_labels
    if labels is None and class_label is not None:
        labels = class_label
    if labels is None:
        labels = sampling_cfg.get("class_labels", sampling_cfg.get("class_label"))
    if labels is None:
        return None
    return class_labels_to_condition(labels, batch_size=batch_size, device=device)


def _label_text(label: int) -> str:
    if 0 <= label < len(CIFAR10_CLASS_NAMES):
        return f"{label}: {CIFAR10_CLASS_NAMES[label]}"
    if label == -1:
        return "-1: null"
    return str(label)


def _images_to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().cpu().float()
    images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError("labeled grids require 1-channel or 3-channel images")
    return (images * 255.0).round().to(torch.uint8)


def save_labeled_image_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    nrow: int = GRID_NROW,
    padding: int = 2,
) -> None:
    """Save a CIFAR10 image grid with one class caption under each sample."""

    images_uint8 = _images_to_uint8(images)
    labels_list = [int(value) for value in labels.detach().cpu().view(-1).tolist()]
    if images_uint8.shape[0] != len(labels_list):
        raise ValueError("number of images and labels must match")

    batch_size, _, height, width = images_uint8.shape
    columns = max(1, min(int(nrow), batch_size))
    rows = math.ceil(batch_size / columns)
    scale = max(1, math.ceil(96 / max(1, min(height, width))))
    tile_width = width * scale
    tile_height = height * scale

    font = ImageFont.load_default()
    probe = Image.new("RGB", (1, 1), "white")
    draw_probe = ImageDraw.Draw(probe)
    label_height = max(
        draw_probe.textbbox((0, 0), _label_text(label), font=font)[3]
        for label in labels_list
    )
    caption_height = label_height + 6

    canvas_width = padding + columns * (tile_width + padding)
    canvas_height = padding + rows * (tile_height + caption_height + padding)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    resampling = getattr(Image, "Resampling", Image).NEAREST
    for index, (image_tensor, label) in enumerate(zip(images_uint8, labels_list, strict=True)):
        row = index // columns
        column = index % columns
        x = padding + column * (tile_width + padding)
        y = padding + row * (tile_height + caption_height + padding)
        image_array = image_tensor.permute(1, 2, 0).numpy()
        tile = Image.fromarray(image_array, mode="RGB")
        if scale != 1:
            tile = tile.resize((tile_width, tile_height), resampling)
        canvas.paste(tile, (x, y))

        text = _label_text(label)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + max(0, (tile_width - text_width) // 2)
        text_y = y + tile_height + 3
        draw.text((text_x, text_y), text, fill="black", font=font)

    canvas.save(output_path)


def checkpoint_ema_state_dict(checkpoint: dict) -> dict:
    """Return mandatory EMA weights from a training checkpoint."""

    if not isinstance(checkpoint, dict) or "model_ema" not in checkpoint:
        raise ValueError(
            "checkpoint must contain model_ema weights. "
            "Retrain with the current EMA-enabled training code.",
        )
    return checkpoint["model_ema"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a diffusion denoiser")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    parser.add_argument("--output", default="samples.png", help="Output image grid")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Force null conditioning for CFG-trained class-conditional configs.",
    )
    parser.add_argument("--class-label", type=int, default=None, help="Single CIFAR10 class label")
    parser.add_argument(
        "--class-labels",
        default=None,
        help="Comma-separated class labels, repeated to fill the batch",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. 1.0 uses the conditional prediction directly.",
    )
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
        model.load_state_dict(checkpoint_ema_state_dict(checkpoint))
        print(f"loaded model_ema from {args.checkpoint}")
    model.eval()
    sampler = build_sampler(config, process, parameterization)

    sampling_cfg = config.get("sampling", {})
    batch_size = args.batch_size or int(sampling_cfg.get("batch_size", 16))
    shape = tuple(sampling_cfg.get("shape", [batch_size, *config.get("data", {}).get("image_shape", [1, 28, 28])]))
    if shape[0] != batch_size:
        shape = (batch_size, *shape[1:])
    condition = _sampling_class_labels(args, config, batch_size, device)
    guidance_scale = (
        float(args.guidance_scale)
        if args.guidance_scale is not None
        else float(sampling_cfg.get("guidance_scale", 1.0))
    )

    samples = sampler.sample(
        model,
        shape=shape,
        condition=condition,
        guidance_scale=guidance_scale,
        device=device,
    )
    images = representation.decode(samples).clamp(-1.0, 1.0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if condition is None:
        save_image(images, output_path, nrow=GRID_NROW, normalize=True, value_range=(-1, 1))
    else:
        save_labeled_image_grid(images, condition, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
