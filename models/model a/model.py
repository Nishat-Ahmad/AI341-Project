"""Model architecture and building utilities."""
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_model(num_classes: int) -> nn.Module:
    """Build ResNet-50 transfer learning model.

    Args:
        num_classes: Number of output classes.

    Returns:
        ResNet-50 model with custom fc head.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace fc layer with custom head
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model
