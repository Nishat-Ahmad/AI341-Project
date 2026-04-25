"""Configuration and constants for damage detector."""
from dataclasses import dataclass
from pathlib import Path

EXPECTED_CLASSES = ["Whole", "Damaged"]
NUM_CLASSES = 2
MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"
IMAGE_SIZE = 224


@dataclass
class TrainConfig:
    """Training configuration."""

    data_dir: Path
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 2
    model_output: Path = Path("weights/model b/best_damage_detector.pth")
    seed: int = 42
    recall_weight: float = 2.0  # Weight for recall optimization
