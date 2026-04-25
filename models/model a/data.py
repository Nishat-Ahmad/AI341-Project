"""Data loading, transforms, and dataset utilities."""
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, EXPECTED_CLASSES


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build training and validation dataloaders.

    Args:
        data_dir: Path to ImageFolder dataset.
        batch_size: Batch size for dataloaders.
        val_split: Fraction of data to use for validation.
        num_workers: Number of dataloader workers.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, class_names).

    Raises:
        ValueError: If classes don't match expected or split fails.
    """
    train_transform, val_transform = get_transforms()

    full_train_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_transform)

    class_names = full_train_dataset.classes
    if set(class_names) != set(EXPECTED_CLASSES):
        raise ValueError(
            "Found classes do not match expected classes. "
            f"Expected: {EXPECTED_CLASSES}, Found: {class_names}"
        )

    indices = list(range(len(full_train_dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if not train_indices or not val_indices:
        raise ValueError(
            "Train/validation split failed. Ensure dataset has enough images and val_split is valid."
        )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_names
