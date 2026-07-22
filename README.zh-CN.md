# Modular Diffusion

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这份代码是我对扩散模型的一次模块化整理。它不打算复刻某个大型模型，而是把
DDPM、DDIM、潜空间扩散和 CFG 分别拆开，放进一套能实际运行、方便替换，也方便做对照实验的实现里。

```text
CIFAR10 图像 / 潜空间噪声 -> 去噪网络 -> 生成图像
```

README 主要讲思路、公式、代码对应关系和实验结果。环境配置、数据下载、训练和采样命令统一放在
[Documentation](Documentation/README.md)。

## 基本想法

扩散模型的前向加噪过程不需要学习，它只负责规定一张干净图像如何一步步退化为噪声。真正需要训练的是反方向的去噪网络：给它一份带噪样本和对应的时间步，让它判断怎样恢复其中的信号。

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
```

这里的关键量是 `alpha_bar`，它决定 `x_t` 中还剩多少原始信号、混入了多少噪声。代码把这部分单独放在
[`diffusion/schedules.py`](diffusion/schedules.py) 和
[`diffusion/processes.py`](diffusion/processes.py)。因此，无论换成线性、余弦还是 sigmoid 噪声调度，都不用改动模型本身。

## 网络到底预测什么

面对同一个带噪样本 `x_t`，网络可以学习不同的预测目标。最常见的是预测噪声，也可以直接预测干净图像 `x0`，或者预测速度 `v`。

```math
\epsilon_\theta(x_t,t), \quad \hat{x}_{0,\theta}(x_t,t), \quad v_\theta(x_t,t)
```

三种写法都来自同一条前向公式，只是参数化方式不同。把目标之间的换算独立出来以后，采样器便不必关心模型训练时采用了哪一种参数化。

对应实现是 [`diffusion/parameterizations.py`](diffusion/parameterizations.py)。

## 训练在做什么

以噪声预测为例，一步训练可以概括为：取一张 CIFAR10 图像，随机选择时间步，加入高斯噪声，再让网络猜出刚刚加入的这份噪声。

```math
\mathcal{L}
= \mathbb{E}_{x_0,t,\epsilon}
\left\|\epsilon - \epsilon_\theta(x_t,t,c)\right\|^2
```

这里的 `c` 表示类别条件。训练时会以一定概率把类别标签换成空条件，这就是 classifier-free guidance（CFG）的训练方式。这样得到的同一个模型既能无条件生成，也能按类别生成，还可以在采样时调节引导强度。

损失函数在 [`diffusion/losses.py`](diffusion/losses.py)，训练入口在
[`diffusion/train.py`](diffusion/train.py)。检查点保存经过预热的 EMA 权重
`model_ema`，采样时读取的也是这一份参数。

## 采样怎么走回图像

采样从纯噪声开始，逐步调用去噪网络。网络负责提供去噪方向，至于每一步具体怎样更新，则由采样器决定。

DDPM 会在每一步重新注入随机噪声：

```math
x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z,
\quad z \sim \mathcal{N}(0,I)
```

DDIM 更多地利用网络恢复出的 `x0` 和噪声方向，因此可以跳过部分时间步，用更少的网络调用完成采样：

```math
x_\tau =
\sqrt{\bar{\alpha}_\tau}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_\tau}\hat{\epsilon}
```

如果加入类别条件，CFG 的做法并不复杂：分别计算无条件预测和有条件预测，再放大二者之间的差异。

```math
\mathrm{pred}
= \mathrm{pred}_{\mathrm{uncond}}
+ s\left(\mathrm{pred}_{\mathrm{cond}}
- \mathrm{pred}_{\mathrm{uncond}}\right)
```

采样器实现都在 [`diffusion/samplers.py`](diffusion/samplers.py)。

## 像素空间和潜空间

像素空间扩散直接对 CIFAR10 的 `3x32x32` 图像张量去噪。潜空间扩散则多了一次表示变换：先用预训练 VAE 把图像编码到潜空间，在其中运行扩散过程，最后再解码回图像。

```text
图像 -> VAE 编码器 -> 潜变量 -> 扩散模型 -> 潜变量 -> VAE 解码器 -> 图像
```

这也是代码里单独保留 `representation` 层的原因。对扩散核心来说，像素和潜变量都只是张量；区别只发生在进入扩散过程之前和离开之后。

相关代码在
[`diffusion/representations/latent.py`](diffusion/representations/latent.py) 和
[`diffusion/models/diffusers_autoencoder.py`](diffusion/models/diffusers_autoencoder.py)。

## 实验覆盖

这里没有穷举所有组合，而是挑选了几组能说明问题的配置。网络结构、噪声调度、预测目标、损失加权和采样器都能找到相应的对照。

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

## 几种做法放在一起看

| 方法 | 学到什么 | 怎么生成 |
| --- | --- | --- |
| DDPM | 每个离散时间步上的去噪预测 | 带随机性的反向链 |
| DDIM | 仍然使用 DDPM 训练出的去噪网络 | 可以跳步，采样更快 |
| 潜空间扩散 | 潜空间中的去噪网络 | 最后把潜变量解码成图像 |
| 这份实现 | 可替换的调度、预测目标、模型和采样器 | 在 CIFAR10 上比较像素空间与潜空间扩散 |

## 结果

下面展示的是 `results/` 中已经完成的实验。条件生成图在每张样本下方标出了对应的 CIFAR10 类别。

### UNet Cosine DDIM

<img src="results/cifar10_unet_cosine.cond.png" alt="UNet cosine DDIM 的 CIFAR10 条件生成结果" width="520">

### DiT DDPM

<img src="results/cifar10_dit_ddpm.cond.png" alt="DiT DDPM 的 CIFAR10 条件生成结果" width="520">

### Latent UNet DDIM

<img src="results/latent_unet_ddim.cond.png" alt="Latent UNet DDIM 的 CIFAR10 条件生成结果" width="520">

### MLP DDPM 失败案例

<img src="results/cifar10_mlp_ddpm.cond.png" alt="MLP DDPM 失败实验的 CIFAR10 条件生成结果" width="520">

把 CIFAR10 图像直接展平后交给 MLP，会丢掉图像中很重要的局部结构先验。

## 小结

这份实现想强调一个很朴素的理解：扩散模型并不是一整块黑盒，而是几个相对独立的部分协同工作。

前向过程规定数据如何变成噪声：

```math
x_t = a_t x_0 + s_t\epsilon
```

网络学习在不同噪声水平下提供去噪信息，采样器再决定如何利用这些信息，从 `x_T` 一步步走回图像。

```math
\mathrm{sample}
= \mathrm{Sampler}\left(\mathrm{Denoiser}_\theta, x_T, c\right)
```
