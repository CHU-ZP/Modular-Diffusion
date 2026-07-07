# Classifier-Free Guidance

Classifier-free guidance (CFG) lets one class-conditional model support three
sampling modes:

- unconditional generation;
- class-conditional generation;
- guided class-conditional generation with a tunable guidance scale.

## Training

CFG training randomly drops labels and replaces them with a learned null class
condition:

```text
with probability dropout_prob:
    condition = null
otherwise:
    condition = class_label
```

In config form:

```yaml
data:
  num_classes: 10

conditioning:
  type: class
  dropout_prob: 0.1
```

The denoiser backbones use an extra learned null embedding. A label of `-1` is
also treated as null conditioning internally.

## Sampling

Unconditional sampling from a CFG-trained checkpoint:

```bash
uv run python -m diffusion.sample \
  --config configs/cifar10_unet_cosine.yaml \
  --checkpoint runs/cifar10_unet_cosine/best_train_loss.pt \
  --device cuda \
  --unconditional \
  --output outputs/cfg_unconditional.png
```

Class-conditional sampling without extra guidance:

```bash
uv run python -m diffusion.sample \
  --config configs/cifar10_unet_cosine.yaml \
  --checkpoint runs/cifar10_unet_cosine/best_train_loss.pt \
  --device cuda \
  --class-label 3 \
  --guidance-scale 1.0 \
  --output outputs/cfg_class_3.png
```

Guided sampling:

```bash
uv run python -m diffusion.sample \
  --config configs/cifar10_unet_cosine.yaml \
  --checkpoint runs/cifar10_unet_cosine/best_train_loss.pt \
  --device cuda \
  --class-labels 0,1,2,3,4,5,6,7,8,9 \
  --guidance-scale 3.0 \
  --output outputs/cfg_guided_grid.png
```

The sampler combines conditional and unconditional predictions as:

```text
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

When `guidance_scale = 1`, the sampler uses the conditional prediction directly.
When `--unconditional` is provided, the sampler ignores default config labels
and uses the learned null condition.

## Configs

All experiment configs now enable CFG training. Each config declares:

```yaml
data:
  num_classes: 10

conditioning:
  type: class
  dropout_prob: 0.1

sampling:
  class_labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  guidance_scale: 3.0
```

The full runner samples `best_train_loss.pt` by default for every experiment:

- `${experiment}.uncond.png` uses `--unconditional`;
- `${experiment}.cond.png` uses `--class-labels` and `--guidance-scale`, and
  includes CIFAR10 label captions under the samples.

Checkpoints include mandatory EMA weights after training. Sampling always loads
`model_ema`; checkpoints without `model_ema` are rejected.
