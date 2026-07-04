"""Small convolutional autoencoder for latent-space experiments."""

from __future__ import annotations

import math

from torch import Tensor, nn


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ConvAutoencoder(nn.Module):
    """A compact image autoencoder with encode/decode methods.

    The default keeps spatial resolution unchanged, which makes it useful as a
    built-in latent wrapper for small examples. Set ``downsample_factor`` to a
    power of two to make lower-resolution latents.
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_channels: int = 4,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        image_channels = int(image_channels)
        latent_channels = int(latent_channels)
        hidden_channels = int(hidden_channels)
        num_res_blocks = int(num_res_blocks)
        downsample_factor = int(downsample_factor)
        if downsample_factor < 1 or downsample_factor & (downsample_factor - 1):
            raise ValueError("downsample_factor must be a positive power of two")

        num_downsamples = int(math.log2(downsample_factor))
        encoder: list[nn.Module] = [
            nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_downsamples):
            encoder.extend(
                [
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.SiLU(),
                ],
            )
        encoder.extend(ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks))
        encoder.append(nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1))

        decoder: list[nn.Module] = [
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        decoder.extend(ResidualConvBlock(hidden_channels) for _ in range(num_res_blocks))
        for _ in range(num_downsamples):
            decoder.extend(
                [
                    nn.ConvTranspose2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.SiLU(),
                ],
            )
        decoder.extend(
            [
                nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ],
        )

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)
