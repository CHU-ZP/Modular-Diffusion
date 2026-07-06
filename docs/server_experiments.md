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

## 3. CIFAR10 Data Source

CIFAR10 is loaded through Hugging Face `datasets` by default:

```yaml
data:
  type: cifar10
  source: huggingface
  hf_dataset: uoft-cs/cifar10
```

The adapter keeps the same training interface as torchvision CIFAR10:

```text
DataLoader batch -> (images, labels)
images shape     -> [batch, 3, 32, 32]
images range     -> [-1, 1]
labels shape     -> [batch]
```

The cache is stored under `./data/huggingface` unless `hf_cache_dir` is set in
the config.

## 4. Smoke Checks

```bash
uv run python -m unittest discover -s tests

uv run python -m diffusion.sample \
  --config configs/cifar10_mlp_ddpm.yaml \
  --device auto \
  --batch-size 4 \
  --unconditional \
  --output outputs/smoke/cifar10_mlp_untrained.png

uv run python -m diffusion.sample \
  --config configs/latent_unet_ddim.yaml \
  --device auto \
  --batch-size 4 \
  --unconditional \
  --output outputs/smoke/latent_unet_untrained.png
```

These images are generated from untrained denoisers. They only validate the
pipeline.

## 5. Small-Run Training

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
from diffusion.conditioning import apply_classifier_free_dropout, class_labels_to_condition
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
condition_dropout = float(config.get("conditioning", {}).get("dropout_prob", 0.0))

model.train()
for step, (images, labels) in zip(range(5), loader):
    clean = representation.encode(images.to(device))
    labels = apply_classifier_free_dropout(labels.to(device), condition_dropout)
    loss = loss_fn(model, clean, condition=labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"step={step} loss={loss.item():.6f}")

Path("runs/small_run").mkdir(parents=True, exist_ok=True)
Path("outputs/small_run").mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "config": config}, "runs/small_run/latent_unet_small.pt")

model.eval()
sampler = build_sampler(config, process, parameterization)
uncond_samples = sampler.sample(model, shape=(4, 4, 4, 4), condition=None, device=device)
uncond_images = representation.decode(uncond_samples).clamp(-1, 1)
save_image(uncond_images, "outputs/small_run/latent_unet_small.uncond.png", normalize=True, value_range=(-1, 1))

cond_labels = class_labels_to_condition("0,1,2,3", batch_size=4, device=device)
cond_samples = sampler.sample(
    model,
    shape=(4, 4, 4, 4),
    condition=cond_labels,
    guidance_scale=float(config.get("sampling", {}).get("guidance_scale", 3.0)),
    device=device,
)
cond_images = representation.decode(cond_samples).clamp(-1, 1)
save_image(cond_images, "outputs/small_run/latent_unet_small.cond.png", normalize=True, value_range=(-1, 1))
PY
```

## 6. Full Runs

To run all formal experiments on physical CUDA devices 1 and 2, with one
training process per GPU at a time:

```bash
./scripts/run_all_experiments_cuda_1_2.sh
```

The script creates two sequential GPU queues:

```text
CUDA 1:
  cifar10_mlp_ddpm
  cifar10_transformer_ddpm
  cifar10_unet_x0_ddpm
  cifar10_unet_cosine

CUDA 2:
  cifar10_unet_ddpm
  cifar10_dit_ddpm
  cifar10_unet_sigmoid_ddpm
  cifar10_unet_snr_cosine
  latent_unet_ddim
```

It also prepares CIFAR10, downloads the VAE if needed, trains each formal config
for 100 epochs, and samples the final checkpoint twice into `outputs/final/`:

```text
${experiment}.uncond.png
${experiment}.cond.png
```

Logs are written to `logs/full_runs/`, including separate conditional and
unconditional sampling logs.

See `docs/experiment_matrix.md` for the full list of covered components.

Override the physical GPU ids if needed:

```bash
GPU_1=0 GPU_2=3 ./scripts/run_all_experiments_cuda_1_2.sh
```

Override the conditional sampling grid and CFG strength if needed:

```bash
CFG_CLASS_LABELS=0,1,2,3,4,5,6,7,8,9 \
CFG_GUIDANCE_SCALE=4.0 \
./scripts/run_all_experiments_cuda_1_2.sh
```

Pixel baselines:

```bash
uv run python -m diffusion.train --config configs/cifar10_mlp_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_transformer_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_dit_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_sigmoid_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_x0_ddpm.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_cosine.yaml --device auto --output-dir runs
uv run python -m diffusion.train --config configs/cifar10_unet_snr_cosine.yaml --device auto --output-dir runs
```

Latent baseline:

```bash
uv run python -m diffusion.train --config configs/latent_unet_ddim.yaml --device auto --output-dir runs
```

The built-in conv autoencoder config remains available only as a smoke check:

```bash
uv run python -m diffusion.train --config configs/latent_conv_autoencoder_smoke.yaml --device auto --output-dir runs
```

Sample trained checkpoints unconditionally and conditionally:

```bash
uv run python -m diffusion.sample \
  --config configs/latent_unet_ddim.yaml \
  --checkpoint runs/latent_unet_ddim/epoch_0100.pt \
  --device auto \
  --unconditional \
  --output outputs/latent_unet_ddim.uncond.png

uv run python -m diffusion.sample \
  --config configs/latent_unet_ddim.yaml \
  --checkpoint runs/latent_unet_ddim/epoch_0100.pt \
  --device auto \
  --class-labels 0,1,2,3,4,5,6,7,8,9 \
  --guidance-scale 3.0 \
  --output outputs/latent_unet_ddim.cond.png
```
