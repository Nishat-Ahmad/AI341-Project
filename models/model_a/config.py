"""Configuration and constants for body type classifier."""
from dataclasses import dataclass
from pathlib import Path

EXPECTED_CLASSES = ["Convertible", "Coupe", "Hatchback", "Pick-Up", "Sedan", "SUV", "VAN"]
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
    model_output: Path = Path("weights/model a/best_body_classifier.pth")
    seed: int = 42
