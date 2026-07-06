#!/usr/bin/env bash
set -euo pipefail

# Run all full CFG-enabled CIFAR10 experiments on physical CUDA devices 1 and 2.
# Each GPU runs one queue sequentially, so a single GPU never hosts more than
# one training process from this script at the same time. Every final checkpoint
# is sampled once unconditionally and once with class guidance.

GPU_1="${GPU_1:-1}"
GPU_2="${GPU_2:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
LOG_DIR="${LOG_DIR:-logs/full_runs}"
SAMPLE_DIR="${SAMPLE_DIR:-outputs/final}"
VAE_DIR="${VAE_DIR:-checkpoints/vae/sd-vae-ft-mse}"
CFG_CLASS_LABELS="${CFG_CLASS_LABELS:-0,1,2,3,4,5,6,7,8,9}"
CFG_GUIDANCE_SCALE="${CFG_GUIDANCE_SCALE:-3.0}"

QUEUE_GPU_1=(
  "configs/cifar10_mlp_ddpm.yaml"
  "configs/cifar10_transformer_ddpm.yaml"
  "configs/cifar10_unet_x0_ddpm.yaml"
  "configs/cifar10_unet_cosine.yaml"
)

QUEUE_GPU_2=(
  "configs/cifar10_unet_ddpm.yaml"
  "configs/cifar10_dit_ddpm.yaml"
  "configs/cifar10_unet_sigmoid_ddpm.yaml"
  "configs/cifar10_unet_snr_cosine.yaml"
  "configs/latent_unet_ddim.yaml"
)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${SAMPLE_DIR}"

experiment_name() {
  uv run python - "$1" <<'PY'
import sys
from diffusion.builders import load_config

print(load_config(sys.argv[1]).get("experiment_name", "diffusion"))
PY
}

final_checkpoint() {
  uv run python - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from diffusion.builders import load_config

config = load_config(sys.argv[1])
output_dir = Path(sys.argv[2])
experiment = config.get("experiment_name", "diffusion")
epochs = int(config.get("training", {}).get("epochs", 1))
print(output_dir / experiment / f"epoch_{epochs:04d}.pt")
PY
}

prepare_cifar10() {
  uv run python - <<'PY'
from diffusion.builders import load_config
from diffusion.train import build_dataloader

config = load_config("configs/cifar10_mlp_ddpm.yaml")
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
next(iter(build_dataloader(config)))
print("CIFAR10 is ready")
PY
}

prepare_vae() {
  if [[ -f "${VAE_DIR}/config.json" ]]; then
    echo "VAE is ready at ${VAE_DIR}"
    return
  fi

  uv run python -m diffusion.download_vae \
    --model-id stabilityai/sd-vae-ft-mse \
    --output-dir "${VAE_DIR}"
}

train_and_sample() {
  local gpu="$1"
  local config="$2"
  local experiment
  local checkpoint
  experiment="$(experiment_name "${config}")"
  checkpoint="$(final_checkpoint "${config}" "${OUTPUT_DIR}")"

  echo "[gpu ${gpu}] training ${experiment}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m diffusion.train \
    --config "${config}" \
    --device cuda \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${experiment}.train.log"

  echo "[gpu ${gpu}] sampling ${experiment} unconditionally from ${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m diffusion.sample \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --device cuda \
    --unconditional \
    --output "${SAMPLE_DIR}/${experiment}.uncond.png" \
    2>&1 | tee "${LOG_DIR}/${experiment}.sample.uncond.log"

  echo "[gpu ${gpu}] sampling ${experiment} conditionally from ${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m diffusion.sample \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --device cuda \
    --class-labels "${CFG_CLASS_LABELS}" \
    --guidance-scale "${CFG_GUIDANCE_SCALE}" \
    --output "${SAMPLE_DIR}/${experiment}.cond.png" \
    2>&1 | tee "${LOG_DIR}/${experiment}.sample.cond.log"
}

run_queue() {
  local gpu="$1"
  shift
  local configs=("$@")

  for config in "${configs[@]}"; do
    train_and_sample "${gpu}" "${config}"
  done
}

prepare_cifar10
prepare_vae

run_queue "${GPU_1}" "${QUEUE_GPU_1[@]}" &
pid_1=$!

run_queue "${GPU_2}" "${QUEUE_GPU_2[@]}" &
pid_2=$!

wait "${pid_1}"
wait "${pid_2}"

echo "All experiments finished."
echo "Logs: ${LOG_DIR}"
echo "Samples: ${SAMPLE_DIR} (*.uncond.png and *.cond.png)"
