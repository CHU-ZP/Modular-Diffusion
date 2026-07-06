# Server Experiments

This checklist prepares a fresh server for CIFAR10 pixel and latent diffusion
experiments.

## 1. Environment

```bash
cd /path/to/Diffusion

uv python install 3.11
uv venv --python 3.11
uv sync
```

`torch` and `torchvision` are resolved from the `pytorch-cu128` uv index on
Linux and Windows, as configured in `pyproject.toml`.

Verify that PyTorch sees the intended device:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## 2. Download VAE

```bash
uv run python -m diffusion.download_vae \
  --model-id stabilityai/sd-vae-ft-mse \
  --output-dir checkpoints/vae/sd-vae-ft-mse
```

The latent config loads this directory with `local_files_only: true`, so training
and sampling do not need network access after the download succeeds.

## 3. Smoke Checks

```bash
uv run python -m unittest discover -s tests

uv run python -m diffusion.sample \
  --config configs/cifar10_mlp_ddpm.yaml \
  --device auto \
  --batch-size 4 \
  --output outputs/smoke/cifar10_mlp_untrained.png

uv run python -m diffusion.sample \
  --config configs/latent_unet_ddim.yaml \
  --device auto \
  --batch-size 4 \
  --output outputs/smoke/latent_unet_untrained.png
```

These images are generated from untrained denoisers. They only validate the
pipeline.

## 4. Small-Run Training

Before long runs, reduce a config in a temporary script and train a few batches:

```bash
uv run python - <<'PY'
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion.builders import (
    build_loss,
    build_model,
    build_optimizer,
    build_parameterization,
    build_process,
    build_representation,
    build_sampler,
    build_schedule,
    load_config,
)
from diffusion.devices import resolve_device
from diffusion.train import build_dataloader

config = load_config("configs/latent_unet_ddim.yaml")
config["schedule"]["num_timesteps"] = 16
config["sampler"]["num_steps"] = 4
config["data"]["batch_size"] = 4
config["data"]["num_workers"] = 0

device = resolve_device("auto")
schedule = build_schedule(config).to(device)
process = build_process(schedule).to(device)
parameterization = build_parameterization(config, schedule)
representation = build_representation(config).to(device)
model = build_model(config).to(device)
loss_fn = build_loss(config, process, parameterization)
optimizer = build_optimizer(config, model)
loader = build_dataloader(config)

model.train()
for step, (images, _) in zip(range(5), loader):
    clean = representation.encode(images.to(device))
    loss = loss_fn(model, clean)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"step={step} loss={loss.item():.6f}")

Path("runs/small_run").mkdir(parents=True, exist_ok=True)
Path("outputs/small_run").mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "config": config}, "runs/small_run/latent_unet_small.pt")

model.eval()
sampler = build_sampler(config, process, parameterization)
samples = sampler.sample(model, shape=(4, 4, 4, 4), device=device)
images = representation.decode(samples).clamp(-1, 1)
save_image(images, "outputs/small_run/latent_unet_small.png", normalize=True, value_range=(-1, 1))
PY
```

## 5. Full Runs

Pixel baselines:

```bash
uv run python -m diffusion.train --config configs/cifar10_mlp_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_cosine.yaml --device auto --output-dir runs
```

Latent baseline:

```bash
uv run python -m diffusion.train --config configs/latent_unet_ddim.yaml --device auto --output-dir runs
```

Sample trained checkpoints:

```bash
uv run python -m diffusion.sample \
  --config configs/latent_unet_ddim.yaml \
  --checkpoint runs/latent_unet_ddim/epoch_0100.pt \
  --device auto \
  --output outputs/latent_unet_ddim.png
```
