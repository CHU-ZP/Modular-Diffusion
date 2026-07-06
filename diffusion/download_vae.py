"""Download a pretrained Diffusers AutoencoderKL for offline experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models.diffusers_autoencoder import load_diffusers_autoencoder_kl, parse_torch_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Diffusers AutoencoderKL")
    parser.add_argument(
        "--model-id",
        default="stabilityai/sd-vae-ft-mse",
        help="Hugging Face model id or local source directory",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/vae/sd-vae-ft-mse",
        help="Directory where the VAE will be saved",
    )
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Dtype used while loading before saving",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_diffusers_autoencoder_kl(
        args.model_id,
        subfolder=args.subfolder,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
    )
    vae.save_pretrained(output_dir, safe_serialization=True)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
