"""CIFAR10 data utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as torchvision_datasets
from torchvision import transforms


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ],
    )


def _canonical_hf_dataset_name(dataset_name: str) -> str:
    if dataset_name == "cifar10":
        return "uoft-cs/cifar10"
    return dataset_name


def _load_huggingface_split(
    dataset_name: str,
    split: str,
    cache_dir: str | Path,
    download: bool,
) -> Any:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face CIFAR10 loading requires the `datasets` package. "
            "Install it with `uv sync` after updating pyproject.toml.",
        ) from exc

    download_config = None
    if not download:
        download_config = DownloadConfig(local_files_only=True)
    return load_dataset(
        _canonical_hf_dataset_name(dataset_name),
        split=split,
        cache_dir=str(cache_dir),
        download_config=download_config,
    )


class HuggingFaceCIFAR10Dataset(Dataset[tuple[Tensor, int]]):
    """Adapter that matches torchvision CIFAR10's `(image_tensor, label)` API."""

    IMAGE_KEYS = ("img", "image")

    def __init__(
        self,
        root: str | Path = "./data",
        train: bool = True,
        download: bool = True,
        transform: transforms.Compose | None = None,
        dataset_name: str = "uoft-cs/cifar10",
        cache_dir: str | Path | None = None,
    ) -> None:
        split = "train" if train else "test"
        cache_dir = Path(cache_dir) if cache_dir is not None else Path(root) / "huggingface"
        self.dataset = _load_huggingface_split(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
            download=download,
        )
        self.transform = _default_transform() if transform is None else transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.dataset[index]
        image = self._image_from_record(record)
        label = int(record["label"])
        return self.transform(image), label

    def _image_from_record(self, record: dict[str, Any]) -> Image.Image:
        for key in self.IMAGE_KEYS:
            if key in record:
                image = record[key]
                break
        else:
            raise KeyError(f"expected one of image keys {self.IMAGE_KEYS}, got {record.keys()}")

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(image).convert("RGB")


def _build_torchvision_cifar10(
    root: str,
    train: bool,
    download: bool,
    transform: transforms.Compose,
) -> Dataset[tuple[Tensor, int]]:
    return torchvision_datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def build_cifar10_dataloader(
    root: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    num_workers: int = 2,
    shuffle: bool | None = None,
    source: str = "huggingface",
    hf_dataset: str = "uoft-cs/cifar10",
    hf_cache_dir: str | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    transform = _default_transform()
    source = source.lower()

    if source in {"huggingface", "hf"}:
        dataset: Dataset[tuple[Tensor, int]] = HuggingFaceCIFAR10Dataset(
            root=root,
            train=train,
            download=download,
            transform=transform,
            dataset_name=hf_dataset,
            cache_dir=hf_cache_dir,
        )
    elif source in {"torchvision", "tv"}:
        dataset = _build_torchvision_cifar10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
    else:
        raise ValueError("source must be one of 'huggingface' or 'torchvision'")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
