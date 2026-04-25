"""Swin model architecture for damage detection."""
from torch import nn
from transformers import AutoModelForImageClassification

from .config import MODEL_ID, NUM_CLASSES


def build_model() -> nn.Module:
    """Build Swin Transformer model for binary damage classification.

    Returns:
        ViT model with custom classification head.
    """
    # Load pre-trained Swin
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # Freeze transformer backbone (works for Swin and other HF image models)
    if hasattr(model, "swin"):
        for param in model.swin.parameters():
            param.requires_grad = False
    elif hasattr(model, "base_model"):
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Only train the classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
