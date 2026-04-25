"""Configuration and constants for body type classifier."""
from dataclasses import dataclass
from pathlib import Path

EXPECTED_CLASSES = ["Sedan", "SUV", "Hatchback", "Convertible", "Coupe"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainConfig:
    """Training configuration."""

    data_dir: Path
    batch_size: int = 32
    val_split: float = 0.2
    epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 2
    model_output: Path = Path("best_body_classifier.pth")
    seed: int = 42
