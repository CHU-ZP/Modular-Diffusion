# Modular Diffusion

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这个项目是我对扩散模型的一次模块化整理。目标不是复刻某个大型模型，而是把
DDPM、DDIM、latent diffusion、CFG 这些概念拆开，放到一套能跑通、能替换、能对照实验的代码里。

```text
CIFAR10 图像 / latent 噪声 -> 去噪网络 -> 生成图像
```

README 只讲思路、公式、代码对应关系和结果。环境配置、数据下载、训练和采样命令放在
[Documentation](Documentation/README.md)。

## 基本想法

扩散模型里，前向加噪过程是不学的。它只是规定：一张干净图像会怎样逐步退化成噪声。真正要训练的是反方向的网络，也就是给定一个带噪样本和时间步，判断应该往哪里去噪。

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
```

这里的关键量是 `alpha_bar`。它控制还保留多少原始信号，以及混入多少噪声。代码里把这部分单独放在
[`diffusion/schedules.py`](diffusion/schedules.py) 和
[`diffusion/processes.py`](diffusion/processes.py)，这样后面换 linear、cosine 或 sigmoid schedule 时，不需要改模型。

## 网络到底预测什么

同一个带噪样本 `x_t`，网络可以学不同的目标。最常见的是预测噪声，也可以直接预测干净图像 `x0`，或者预测 velocity。

```math
\epsilon_\theta(x_t,t), \quad \hat{x}_{0,\theta}(x_t,t), \quad v_\theta(x_t,t)
```

这三种目标本质上是同一条前向公式的不同重排。把它们抽成独立模块之后，sampler 就不用关心训练时选的是哪一种目标。

对应实现是 [`diffusion/parameterizations.py`](diffusion/parameterizations.py)。

## 训练在做什么

以噪声预测为例，每一步训练都很直接：拿一张 CIFAR10 图像，随机选一个时间步，加一份高斯噪声，然后让网络把这份噪声预测出来。

```math
\mathcal{L}
= \mathbb{E}_{x_0,t,\epsilon}
\left\|\epsilon - \epsilon_\theta(x_t,t,c)\right\|^2
```

这里的 `c` 是类别条件。训练时会按一定概率把类别标签替换成 null condition，这就是 classifier-free guidance 的训练方式。这样训练完的同一个模型，可以无条件生成，也可以按类别生成，还可以调 guidance scale。

loss 在 [`diffusion/losses.py`](diffusion/losses.py)，训练入口在
[`diffusion/train.py`](diffusion/train.py)。checkpoint 里保存的是 warmup EMA 后的
`model_ema`，采样时也只读取这份权重。

## 采样怎么走回图像

采样时从纯噪声开始，一步步调用网络。网络只负责给出去噪信息，真正决定“怎么走”的是 sampler。

DDPM 会在每一步重新注入随机噪声：

```math
x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z,
\quad z \sim \mathcal{N}(0,I)
```

DDIM 则更多地使用网络恢复出的 `x0` 和噪声方向，可以用更少的步数跳着走：

```math
x_\tau =
\sqrt{\bar{\alpha}_\tau}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_\tau}\hat{\epsilon}
```

如果使用类别条件，CFG 做的事情也很朴素：同时看一次无条件预测和有条件预测，再把二者的差放大。

```math
\mathrm{pred}
= \mathrm{pred}_{\mathrm{uncond}}
+ s\left(\mathrm{pred}_{\mathrm{cond}}
- \mathrm{pred}_{\mathrm{uncond}}\right)
```

采样器实现都在 [`diffusion/samplers.py`](diffusion/samplers.py)。

## Pixel 和 Latent

pixel diffusion 直接在 CIFAR10 的 `3x32x32` 图像空间里去噪。latent diffusion 多了一层表示变换：先用预训练 VAE 把图像编码成 latent，在 latent 里训练扩散模型，最后再解码回图像。

```text
image -> VAE encoder -> latent -> diffusion -> latent -> VAE decoder -> image
```

这也是我保留 `representation` 这一层抽象的原因。pixel 和 latent 对 diffusion core 来说都是 tensor，只是进入扩散过程之前和离开扩散过程之后的表示不同。

相关代码在
[`diffusion/representations/latent.py`](diffusion/representations/latent.py) 和
[`diffusion/models/diffusers_autoencoder.py`](diffusion/models/diffusers_autoencoder.py)。

## 实验覆盖

实验没有做成巨大的组合网格，而是挑了几组能说明问题的配置：网络、schedule、预测目标、loss weighting、sampler 都至少有对应的对照。

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

模型在 [`diffusion/models/`](diffusion/models/)，配置到组件的构建逻辑在
[`diffusion/builders.py`](diffusion/builders.py)。

## 一点对照

| 方法 | 学到什么 | 怎么生成 |
| --- | --- | --- |
| DDPM | 每个离散时间步上的去噪预测 | 带随机性的反向链 |
| DDIM | 仍然使用 DDPM 训练出的去噪网络 | 可以跳步，采样更快 |
| Latent diffusion | latent 空间里的去噪网络 | 最后把 latent 解码成图像 |
| 这个项目 | 可替换的 schedule、target、model、sampler | 在 CIFAR10 上做 pixel 和 latent 实验 |

## 结果

这些图是 `results/` 里的已完成实验结果。条件生成图下面带有 CIFAR10 类别标签。

### UNet Cosine DDIM

<img src="results/cifar10_unet_cosine.cond.png" alt="UNet cosine DDIM 的 CIFAR10 条件生成结果" width="520">

### DiT DDPM

<img src="results/cifar10_dit_ddpm.cond.png" alt="DiT DDPM 的 CIFAR10 条件生成结果" width="520">

### Latent UNet DDIM

<img src="results/latent_unet_ddim.cond.png" alt="Latent UNet DDIM 的 CIFAR10 条件生成结果" width="520">

### MLP DDPM 失败案例

<img src="results/cifar10_mlp_ddpm.cond.png" alt="MLP DDPM 失败实验的 CIFAR10 条件生成结果" width="520">

把 CIFAR10 直接 flatten 后交给 MLP，会丢掉图像里很重要的局部结构先验。

## 小结

我希望这个仓库表达的是一个比较朴素的理解：扩散模型不是一个单块黑盒，而是几件事的配合。

前向过程规定数据如何变成噪声：

```math
x_t = a_t x_0 + s_t\epsilon
```

网络学习在每个噪声水平下提供去噪信息。sampler 再决定如何利用这些信息，从 `x_T` 走回图像。

```math
\mathrm{sample}
= \mathrm{Sampler}\left(\mathrm{Denoiser}_\theta, x_T, c\right)
```
