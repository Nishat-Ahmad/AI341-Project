"""Data loading and augmentation utilities."""
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from .config import EXPECTED_CLASSES, IMAGE_SIZE, MODEL_ID


class DamagedImageAugmentation:
    """Heavy augmentation for Damaged class to simulate different angles."""

    def __call__(self, img):
        """Apply random augmentations."""
        img = transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0))(img)
        img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)(img)
        img = transforms.RandomRotation(25)(img)
        img = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))(img)
        img = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))(img)
        return img


class DamageDataset(Dataset):
    """Custom dataset for damage detection with class-specific augmentation."""

    def __init__(
        self,
        data_dir: Path,
        processor,
        is_train: bool = True,
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to dataset root (parent of Whole/Damaged folders).
            processor: Hugging Face image processor for preprocessing.
            is_train: Whether this is training set.
        """
        self.processor = processor
        self.is_train = is_train
        self.images = []
        self.labels = []

        # Load images and labels
        for class_idx, class_name in enumerate(EXPECTED_CLASSES):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            for img_path in class_dir.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(class_idx)

            for img_path in class_dir.glob("*.png"):
                self.images.append(img_path)
                self.labels.append(class_idx)

        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

        # Drop stale paths if files were moved/deleted after listing.
        filtered = [(p, y) for p, y in zip(self.images, self.labels) if p.exists() and p.is_file()]
        self.images = [p for p, _ in filtered]
        self.labels = [y for _, y in filtered]

        if not self.images:
            raise ValueError(f"No readable image files found in {data_dir}")

        # Augmentation for training
        self.damaged_aug = DamagedImageAugmentation() if is_train else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and label.

        Args:
            idx: Index of image.

        Returns:
            Tuple of (image_tensor, label_tensor).
        """
        from PIL import Image

        max_tries = min(16, len(self.images))
        for offset in range(max_tries):
            safe_idx = (idx + offset) % len(self.images)
            img_path = self.images[safe_idx]
            label = self.labels[safe_idx]

            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                continue

            # Apply heavy augmentation to Damaged class during training
            if self.is_train and label == 1:  # Damaged class
                image = self.damaged_aug(image)

            # Process with ViTImageProcessor
            processed = self.processor(image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            return pixel_values, torch.tensor(label, dtype=torch.long)

        raise FileNotFoundError("Failed to load a valid image sample after multiple attempts.")


class BinaryDamageDataset(Dataset):
    """Binary dataset built from explicit (image_path, label) pairs."""

    def __init__(
        self,
        samples: List[Tuple[Path, int, bool]],
        processor,
        is_train: bool = True,
    ):
        self.samples = [(p, y, syn) for p, y, syn in samples if p.exists() and p.is_file()]
        self.processor = processor
        self.is_train = is_train
        self.damaged_aug = DamagedImageAugmentation() if is_train else None
        self.synthetic_negative_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 0.6)),
                transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
                transforms.RandomGrayscale(p=0.4),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5)),
            ]
        )

        if not self.samples:
            raise ValueError("No valid image files found after filtering sample paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        max_tries = min(16, len(self.samples))
        for offset in range(max_tries):
            safe_idx = (idx + offset) % len(self.samples)
            img_path, label, is_synthetic_negative = self.samples[safe_idx]

            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                continue

            if self.is_train and label == 1:  # Damaged class
                image = self.damaged_aug(image)
            elif is_synthetic_negative:
                # Damaged-only fallback: synthesize non-damage-like negatives.
                image = self.synthetic_negative_aug(image)

            processed = self.processor(image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            return pixel_values, torch.tensor(label, dtype=torch.long)

        raise FileNotFoundError("Failed to load a valid image sample after multiple attempts.")


def _collect_images(folder: Path) -> List[Path]:
    """Collect images from a folder recursively."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.rglob(ext))
    return sorted(files)


def _split_items(items: List[Path], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """Deterministically split items into train/validation."""
    if len(items) < 2:
        return items, []

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(items), generator=g).tolist()
    split_at = max(1, int(len(items) * (1.0 - val_ratio)))
    train_idx = perm[:split_at]
    val_idx = perm[split_at:]

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items


def _resolve_damage_assessment_root(data_dir: Path) -> Path | None:
    """Resolve CarDD/FiftyOne-style dataset root."""
    if (data_dir / "samples.json").exists() and (data_dir / "data").exists():
        return data_dir

    nested = data_dir / "damage_assessment"
    if (nested / "samples.json").exists() and (nested / "data").exists():
        return nested

    return None


def _build_from_damage_assessment(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    processor,
) -> Tuple[DataLoader, DataLoader, object]:
    """Build loaders from CarDD/FiftyOne-style layout.

    Expected positives: <root>/damage_assessment/data or <root>/data
    Expected negatives: any of
      - <data_dir>/whole_pool
      - <data_dir>/Whole
      - <data_dir>/train/Whole + <data_dir>/valid/Whole
    """
    assessment_root = _resolve_damage_assessment_root(data_dir)
    if assessment_root is None:
        raise ValueError("Could not resolve damage_assessment dataset root.")

    damaged_images = _collect_images(assessment_root / "data")
    if not damaged_images:
        raise ValueError(f"No images found in {assessment_root / 'data'}")

    whole_candidates = [
        data_dir / "whole_pool",
        data_dir / "Whole",
        assessment_root.parent / "whole_pool",
        assessment_root.parent / "Whole",
    ]

    whole_images: List[Path] = []
    for folder in whole_candidates:
        if folder.exists():
            whole_images.extend(_collect_images(folder))

    train_whole = data_dir / "train" / "Whole"
    valid_whole = data_dir / "valid" / "Whole"
    parent_train_whole = assessment_root.parent / "train" / "Whole"
    parent_valid_whole = assessment_root.parent / "valid" / "Whole"
    if train_whole.exists():
        whole_images.extend(_collect_images(train_whole))
    if valid_whole.exists():
        whole_images.extend(_collect_images(valid_whole))
    if parent_train_whole.exists():
        whole_images.extend(_collect_images(parent_train_whole))
    if parent_valid_whole.exists():
        whole_images.extend(_collect_images(parent_valid_whole))

    whole_images = sorted(set(whole_images))
    train_damaged, val_damaged = _split_items(damaged_images, val_ratio=0.2, seed=42)
    train_samples: List[Tuple[Path, int, bool]]
    val_samples: List[Tuple[Path, int, bool]]

    if whole_images:
        train_whole_split, val_whole_split = _split_items(whole_images, val_ratio=0.2, seed=42)
        train_samples = [(p, 1, False) for p in train_damaged] + [(p, 0, False) for p in train_whole_split]
        val_samples = [(p, 1, False) for p in val_damaged] + [(p, 0, False) for p in val_whole_split]
    else:
        print(
            "WARNING: No Whole images found. Using damaged-only fallback with synthetic Whole negatives."
        )
        # Each damaged image is used twice: once as Damaged, once as synthetic Whole.
        train_samples = [(p, 1, False) for p in train_damaged] + [(p, 0, True) for p in train_damaged]
        val_samples = [(p, 1, False) for p in val_damaged] + [(p, 0, True) for p in val_damaged]

    if not train_samples or not val_samples:
        raise ValueError("Insufficient samples after split. Need at least one train and one validation sample per run.")

    g = torch.Generator().manual_seed(42)
    train_perm = torch.randperm(len(train_samples), generator=g).tolist()
    val_perm = torch.randperm(len(val_samples), generator=g).tolist()
    train_samples = [train_samples[i] for i in train_perm]
    val_samples = [val_samples[i] for i in val_perm]

    train_dataset = BinaryDamageDataset(train_samples, processor, is_train=True)
    val_dataset = BinaryDamageDataset(val_samples, processor, is_train=False)

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

    return train_loader, val_loader, processor


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, object]:
    """Build train and validation dataloaders.

    Args:
        data_dir: Path to dataset root.
        batch_size: Batch size.
        num_workers: Number of workers.

    Returns:
        Tuple of (train_loader, val_loader, processor).
    """
    # Load processor
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    # Layout A: split folders (existing format)
    train_dir = data_dir / "train"
    val_dir = data_dir / "valid"

    if train_dir.exists() and val_dir.exists():
        train_dataset = DamageDataset(train_dir, processor, is_train=True)
        val_dataset = DamageDataset(val_dir, processor, is_train=False)

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

        return train_loader, val_loader, processor

    # Layout B: CarDD/FiftyOne damage_assessment format
    assessment_root = _resolve_damage_assessment_root(data_dir)
    if assessment_root is not None:
        return _build_from_damage_assessment(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            processor=processor,
        )

    raise ValueError(
        f"Unsupported dataset layout in {data_dir}. Expected either train/valid folders "
        "or a damage_assessment dataset with data/ and samples.json."
    )
