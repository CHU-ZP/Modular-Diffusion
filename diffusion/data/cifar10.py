"""CIFAR10 data utilities."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_dataloader(
    root: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    num_workers: int = 2,
    shuffle: bool | None = None,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ],
    )
    dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
