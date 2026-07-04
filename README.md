# Modular Diffusion

A small diffusion research playground with stable math interfaces and replaceable
components:

- schedules: linear, cosine, sigmoid
- prediction targets: epsilon, x0, v
- samplers: DDPM, DDIM
- backbones: MLP, UNet, Transformer, DiT with AdaLN-Zero blocks
- representations: pixel identity and configurable latent autoencoder wrapper

Run smoke tests:

```bash
python -m unittest discover -s tests
```

Train a tiny MNIST baseline:

```bash
python -m diffusion.train --config configs/mnist_mlp_ddpm.yaml --device auto
```

Sample from a checkpoint:

```bash
python -m diffusion.sample --config configs/mnist_mlp_ddpm.yaml --checkpoint runs/mnist_mlp_ddpm/epoch_0001.pt --output samples.png
```
