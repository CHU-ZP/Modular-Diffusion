# Documentation

This page contains the practical commands for setting up the environment,
preparing data, training, sampling, and checking outputs.

## Environment

```bash
cd /path/to/Diffusion

uv python install 3.11
uv venv --python 3.11
uv sync
```

The project uses `uv`. On Linux and Windows, `torch` and `torchvision` resolve
from the CUDA 12.8 PyTorch index configured in [`pyproject.toml`](../pyproject.toml).

Check CUDA:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("num_gpus:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## Prepare the VAE

The latent experiment expects a local Diffusers VAE checkpoint:

```bash
uv run python -m diffusion.download_vae \
  --model-id stabilityai/sd-vae-ft-mse \
  --output-dir checkpoints/vae/sd-vae-ft-mse
```

The latent config loads this path with `local_files_only: true`.

## Prepare CIFAR10

CIFAR10 is loaded through Hugging Face `datasets` by default:

```yaml
data:
  type: cifar10
  source: huggingface
  hf_dataset: uoft-cs/cifar10
```

The adapter keeps the training interface compatible with torchvision CIFAR10:

```text
DataLoader batch -> (images, labels)
images shape     -> [batch, 3, 32, 32]
images range     -> [-1, 1]
labels shape     -> [batch]
```

Optionally trigger the download before training:

```bash
uv run python - <<'PY'
from diffusion.builders import load_config
from diffusion.train import build_dataloader

config = load_config("configs/cifar10_mlp_ddpm.yaml")
config["data"]["batch_size"] = 4
config["data"]["num_workers"] = 0

images, labels = next(iter(build_dataloader(config)))
print(images.shape, images.min().item(), images.max().item())
print(labels[:4])
PY
```

## Smoke Checks

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

Untrained samples only validate that the pipeline runs.

## Full Training

Run all formal experiments on two physical CUDA devices:

```bash
GPU_1=1 GPU_2=2 \
OUTPUT_DIR=runs \
LOG_DIR=logs/full_runs_cuda12 \
SAMPLE_DIR=outputs/final_cuda12 \
./scripts/run_all_experiments_cuda_1_2.sh
```

For CUDA 0 and 1:

```bash
GPU_1=0 GPU_2=1 \
OUTPUT_DIR=runs \
LOG_DIR=logs/full_runs_cuda01 \
SAMPLE_DIR=outputs/final_cuda01 \
./scripts/run_all_experiments_cuda_1_2.sh
```

If the script is not executable:

```bash
bash scripts/run_all_experiments_cuda_1_2.sh
```

The runner trains these eight formal experiments:

```text
cifar10_mlp_ddpm
cifar10_unet_ddpm
cifar10_dit_ddpm
cifar10_unet_sigmoid_ddpm
cifar10_unet_x0_ddpm
cifar10_unet_cosine
cifar10_unet_snr_cosine
latent_unet_ddim
```

Each config trains for 100 epochs and saves warmup EMA weights under
`model_ema`.

## Run in tmux

```bash
tmux new -s diffusion
```

Inside the session:

```bash
cd /path/to/Diffusion

GPU_1=1 GPU_2=2 \
OUTPUT_DIR=runs \
LOG_DIR=logs/full_runs_cuda12 \
SAMPLE_DIR=outputs/final_cuda12 \
./scripts/run_all_experiments_cuda_1_2.sh
```

Detach without stopping training:

```text
Ctrl-b then d
```

Reattach:

```bash
tmux attach -t diffusion
```

## Output Locations

Checkpoints:

```text
runs/<experiment>/last.pt
runs/<experiment>/best_train_loss.pt
runs/<experiment>/epoch_0005.pt
runs/<experiment>/epoch_0010.pt
...
runs/<experiment>/epoch_0100.pt
```

Logs:

```text
logs/full_runs_cuda12/<experiment>.train.log
logs/full_runs_cuda12/<experiment>.sample.uncond.log
logs/full_runs_cuda12/<experiment>.sample.cond.log
```

Generated images:

```text
outputs/final_cuda12/<experiment>.uncond.png
outputs/final_cuda12/<experiment>.cond.png
```

Training log lines include `ema_decay`, so EMA warmup can be checked with:

```bash
tail -f logs/full_runs_cuda12/cifar10_unet_cosine.train.log
```

## Sampling Existing Checkpoints

Unconditional:

```bash
uv run python -m diffusion.sample \
  --config configs/cifar10_unet_cosine.yaml \
  --checkpoint runs/cifar10_unet_cosine/best_train_loss.pt \
  --device cuda \
  --unconditional \
  --output outputs/check_unet_cosine_uncond.png
```

Class-conditional with CFG:

```bash
uv run python -m diffusion.sample \
  --config configs/cifar10_unet_cosine.yaml \
  --checkpoint runs/cifar10_unet_cosine/best_train_loss.pt \
  --device cuda \
  --class-labels 0,1,2,3,4,5,6,7,8,9 \
  --guidance-scale 3.0 \
  --output outputs/check_unet_cosine_cond.png
```

Sampling always loads `model_ema`. Checkpoints without `model_ema` are rejected.

## Sampling a Different Checkpoint

The full runner samples `best_train_loss.pt` by default. To sample `last.pt`:

```bash
CHECKPOINT_NAME=last.pt \
GPU_1=1 GPU_2=2 \
OUTPUT_DIR=runs \
LOG_DIR=logs/full_runs_cuda12_last \
SAMPLE_DIR=outputs/final_cuda12_last \
./scripts/run_all_experiments_cuda_1_2.sh
```

To sample the final periodic checkpoint, `epoch_0100.pt`:

```bash
CHECKPOINT_NAME=final \
GPU_1=1 GPU_2=2 \
OUTPUT_DIR=runs \
LOG_DIR=logs/full_runs_cuda12_final \
SAMPLE_DIR=outputs/final_cuda12_final \
./scripts/run_all_experiments_cuda_1_2.sh
```

These runner commands retrain before sampling. To resample an already trained
checkpoint, run `diffusion.sample` directly.
