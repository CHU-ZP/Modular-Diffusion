# Modular Diffusion

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这是一个面向理解的扩散模型仓库：用尽量清楚的 PyTorch 模块，把 DDPM、
DDIM、latent diffusion、classifier-free guidance 这些核心概念串起来。

```text
CIFAR10 图像或 latent 噪声 -> 去噪网络 -> 生成的 CIFAR10 图像
```

这个 README 主要解释项目背后的想法、公式和代码实现对应关系，并展示已经跑出的结果。环境配置、数据准备、训练和采样命令放在
[Documentation](Documentation/README.md)。

## 直觉

扩散模型先规定一个固定的加噪过程，把干净样本一步步推向高斯噪声；训练时，网络学习在任意噪声水平下把“该去掉什么”预测出来。真正需要学习的是去噪网络，而不是前向加噪过程。

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
```

噪声调度决定信号衰减的速度。本仓库实现了 linear、cosine、sigmoid 三种调度，对应代码在
[`diffusion/schedules.py`](diffusion/schedules.py)。前向加噪和后验系数在
[`diffusion/processes.py`](diffusion/processes.py)。

## 从图像到噪声

训练时会采样一张干净图像 `x0`、一个时间步 `t` 和一份高斯噪声
`epsilon`。网络看到的是带噪图像 `x_t` 和时间步，它可以预测三类等价目标：噪声、干净样本，或者 velocity。

```math
\epsilon\text{-target}: \epsilon_\theta(x_t,t)
```

```math
x_0\text{-target}: \hat{x}_{0,\theta}(x_t,t)
```

```math
v\text{-target}: v_\theta(x_t,t)
```

这些目标之间的转换写在
[`diffusion/parameterizations.py`](diffusion/parameterizations.py)。这样 sampler
不用关心模型训练时到底预测的是哪一种目标。

## 训练到底在学什么

以最经典的噪声预测为例，训练目标就是让网络预测出当初加进去的那份噪声：

```math
\mathcal{L}
= \mathbb{E}_{x_0,t,\epsilon}
\left\|\epsilon - \epsilon_\theta(x_t,t,c)\right\|^2
```

这里的 `c` 是可选类别条件。本仓库使用 CIFAR10 类别做 classifier-free
guidance：训练时会随机把一部分标签替换成 learned null condition。这样同一个 checkpoint
既能做无条件生成，也能做类别条件生成，还能调 guidance scale。

loss 封装在 [`diffusion/losses.py`](diffusion/losses.py)，训练循环在
[`diffusion/train.py`](diffusion/train.py)。训练保存的是 warmup EMA 后的
`model_ema` 权重，采样时也强制读取这份权重。

## 怎么生成

生成从纯高斯噪声开始，sampler 在每个时间步调用去噪网络，把网络预测转化成更低噪声水平的样本。DDPM 是随机反向链，DDIM 则可以用更少步数走一条确定性或近似确定性的轨迹。

```math
x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z,
\quad z \sim \mathcal{N}(0,I)
```

```math
x_\tau =
\sqrt{\bar{\alpha}_\tau}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_\tau}\hat{\epsilon}
```

有类别条件时，CFG 会把无条件预测和有条件预测组合起来：

```math
\mathrm{pred}
= \mathrm{pred}_{\mathrm{uncond}}
+ s\left(\mathrm{pred}_{\mathrm{cond}}
- \mathrm{pred}_{\mathrm{uncond}}\right)
```

采样器在 [`diffusion/samplers.py`](diffusion/samplers.py)。

## Pixel Diffusion 和 Latent Diffusion

pixel 实验直接在 `3x32x32` 的 CIFAR10 图像空间里去噪。latent 实验先用预训练
Diffusers `AutoencoderKL` 把图像压到 latent，再在 latent 空间里做扩散，最后解码回图像。

```text
image -> VAE encoder -> latent -> diffusion denoiser -> latent -> VAE decoder -> image
```

latent 表示的封装在
[`diffusion/representations/latent.py`](diffusion/representations/latent.py)，Diffusers
VAE 适配器在
[`diffusion/models/diffusers_autoencoder.py`](diffusion/models/diffusers_autoencoder.py)。

## 和相关方法的关系

| 方法 | 学到的对象 | 生成方式 |
| --- | --- | --- |
| DDPM | 离散时间步上的噪声或数据预测 | 随机反向马尔可夫链 |
| DDIM | 和 DDPM 相同的去噪网络 | 可以跳步的确定性或低随机性采样 |
| Latent diffusion | latent 空间里的去噪模型 | 最终 latent 再解码成图像 |
| 本仓库 | 可替换的网络、调度、目标和采样器 | CIFAR10 pixel 与 latent 生成 |

## 这个项目实现了什么

正式实验都使用 CIFAR10，并启用 classifier-free class conditioning。配置覆盖了不同的网络、调度、预测目标、loss weighting 和采样器。

| 配置 | 空间 | 网络 | 调度 | 目标 | 采样器 |
| --- | --- | --- | --- | --- | --- |
| [`configs/cifar10_mlp_ddpm.yaml`](configs/cifar10_mlp_ddpm.yaml) | pixel | MLP | linear | epsilon | DDPM |
| [`configs/cifar10_unet_ddpm.yaml`](configs/cifar10_unet_ddpm.yaml) | pixel | UNet | linear | epsilon | DDPM |
| [`configs/cifar10_dit_ddpm.yaml`](configs/cifar10_dit_ddpm.yaml) | pixel | DiT | linear | epsilon | DDPM |
| [`configs/cifar10_unet_sigmoid_ddpm.yaml`](configs/cifar10_unet_sigmoid_ddpm.yaml) | pixel | UNet | sigmoid | epsilon | DDPM |
| [`configs/cifar10_unet_x0_ddpm.yaml`](configs/cifar10_unet_x0_ddpm.yaml) | pixel | UNet | linear | x0 | DDPM |
| [`configs/cifar10_unet_cosine.yaml`](configs/cifar10_unet_cosine.yaml) | pixel | UNet | cosine | epsilon | DDIM |
| [`configs/cifar10_unet_snr_cosine.yaml`](configs/cifar10_unet_snr_cosine.yaml) | pixel | UNet | cosine | epsilon | DDIM |
| [`configs/latent_unet_ddim.yaml`](configs/latent_unet_ddim.yaml) | latent | UNet | cosine | v | DDIM |

模型代码在 [`diffusion/models/`](diffusion/models/)，组件构建逻辑集中在
[`diffusion/builders.py`](diffusion/builders.py)。

## 结果展示

下面的图片来自 [`results/`](results) 中已经跑完的实验。条件生成图会在每个 tile
下面标出 CIFAR10 类别。

### UNet Cosine DDIM

<img src="results/cifar10_unet_cosine.cond.png" alt="UNet cosine DDIM 的 CIFAR10 条件生成结果" width="520">

### DiT DDPM

<img src="results/cifar10_dit_ddpm.cond.png" alt="DiT DDPM 的 CIFAR10 条件生成结果" width="520">

### Latent UNet DDIM

<img src="results/latent_unet_ddim.cond.png" alt="Latent UNet DDIM 的 CIFAR10 条件生成结果" width="520">

### MLP DDPM Baseline

<img src="results/cifar10_mlp_ddpm.cond.png" alt="MLP DDPM baseline 的 CIFAR10 条件生成结果" width="520">

## 总结

这个仓库的重点不是追求单一最强配置，而是把扩散模型拆成清楚的组件：前向过程定义数据如何变成噪声，网络学习在不同噪声水平下提供去噪信息，采样器决定如何把这些信息一步步转回图像。

```math
x_t = a_t x_0 + s_t\epsilon
```

```math
\mathrm{sample}
= \mathrm{Sampler}\left(\mathrm{Denoiser}_\theta, x_T, c\right)
```
