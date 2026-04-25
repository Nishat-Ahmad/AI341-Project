"""Data loading, transforms, and dataset utilities."""
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
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
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build training, validation, and test dataloaders from pre-split folders.

    Expected structure:
        data_dir/
            train/
                class1/
                class2/
                ...
            valid/
                class1/
                class2/
                ...
            test/
                class1/
                class2/
                ...

    Args:
        data_dir: Path to dataset root (parent of train/valid/test).
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).

    Raises:
        ValueError: If expected folders don't exist.
    """
    train_transform, val_transform = get_transforms()

    train_dir = data_dir / "train"
    val_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise ValueError(
            f"Expected train/valid/test folders in {data_dir}. "
            f"Found: train={train_dir.exists()}, valid={val_dir.exists()}, test={test_dir.exists()}"
        )

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_transform)

    class_names = train_dataset.classes
    if set(class_names) != set(EXPECTED_CLASSES):
        raise ValueError(
            "Found classes do not match expected classes. "
            f"Expected: {EXPECTED_CLASSES}, Found: {class_names}"
        )

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_names
