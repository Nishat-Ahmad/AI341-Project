"""Inference utilities for car type classification."""
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn

from .config import EXPECTED_CLASSES
from .data import get_transforms
from .model import build_model


def load_trained_model(model_path: Path, device: torch.device) -> Tuple[nn.Module, list[str]]:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        device: Torch device.

    Returns:
        Tuple of (model, class_names).
    """
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", EXPECTED_CLASSES)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def classify_car_type(
    image_path: str,
    model_path: str = "weights/model a/best_body_classifier.pth",
) -> Tuple[str, float]:
    """Classify car body type from image.

    Args:
        image_path: Path to car image.
        model_path: Path to trained model checkpoint.

    Returns:
        Tuple of (predicted_class_name, confidence_score).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_trained_model(Path(model_path), device)

    _, val_transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence_score = confidence.item()
    return predicted_class, confidence_score
