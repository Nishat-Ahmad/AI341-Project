"""Single image inference for damage detection."""
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor

from .config import MODEL_ID


def load_trained_model(
    model_path: Path,
    device: torch.device,
):
    """Load trained image classification model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        device: Torch device.

    Returns:
        Tuple of (model, processor).
    """
    from .model import build_model

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    return model, processor


def classify_damage(
    image_path: str,
    model_path: str = "weights/model b/best_damage_detector.pth",
    device: str | None = None,
) -> Tuple[str, float]:
    """Classify if car is damaged.

    Args:
        image_path: Path to car image.
        model_path: Path to trained model.
        device: Device to use ('cuda', 'cpu', or None for auto with fallback).

    Returns:
        Tuple of (class_name, confidence).
    """
    # Device selection with fallback
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    try:
        model, processor = load_trained_model(Path(model_path), device_obj)
    except RuntimeError as e:
        if "CUDA" in str(e) and device_obj.type == "cuda":
            print("WARNING: CUDA error during model loading. Falling back to CPU...")
            device_obj = torch.device("cpu")
            model, processor = load_trained_model(Path(model_path), device_obj)
        else:
            raise

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    processed = processor(image, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device_obj)

    try:
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
    except RuntimeError as e:
        if "CUDA" in str(e) and device_obj.type == "cuda":
            print("WARNING: CUDA error during inference. Retrying on CPU...")
            device_obj = torch.device("cpu")
            model.to(device_obj)
            pixel_values = pixel_values.to(device_obj)
            with torch.no_grad():
                outputs = model(pixel_values)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
        else:
            raise

    class_name = "Damaged" if pred_idx.item() == 1 else "Whole"
    confidence_score = confidence.item()

    return class_name, confidence_score
