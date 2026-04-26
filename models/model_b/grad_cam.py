"""Grad-CAM implementation for damage localization."""
from pathlib import Path
from typing import Tuple
from datetime import datetime, UTC
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from .config import MODEL_ID


class GradCAMViT:
    """Grad-CAM for transformer image classification models."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        device: torch.device,
        target_layer: str = "auto",
    ):
        """Initialize Grad-CAM.

        Args:
            model: Transformer image model.
            processor: Hugging Face image processor.
            device: Torch device.
            target_layer: Layer to target for CAM.
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: str) -> None:
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Auto-detect a suitable final normalization layer across backbones.
        if target_layer == "auto":
            layer = None
            if hasattr(self.model, "swin") and hasattr(self.model.swin, "layernorm"):
                layer = self.model.swin.layernorm
            elif hasattr(self.model, "vit") and hasattr(self.model.vit, "layernorm"):
                layer = self.model.vit.layernorm
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "layernorm"):
                layer = self.model.base_model.layernorm

            if layer is None:
                raise ValueError("Could not find a supported layernorm target for Grad-CAM.")

            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)
            return

        if "layernorm" in target_layer and hasattr(self.model, "vit"):
            self.model.vit.layernorm.register_forward_hook(forward_hook)
            self.model.vit.layernorm.register_full_backward_hook(backward_hook)

    @torch.enable_grad()
    def generate_cam(
        self,
        image_path: str,
        target_class: int = 1,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            image_path: Path to image.
            target_class: Target class (1 for Damaged).

        Returns:
            Heatmap numpy array.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        processed = self.processor(image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device).requires_grad_(True)

        # Forward pass
        self.model.eval()
        outputs = self.model(pixel_values)
        logits = outputs.logits

        # Backward pass
        target_score = logits[0, target_class]
        self.model.zero_grad()
        target_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0].mean(dim=0)
        activations = self.activations[0]

        # Weighted sum of activations
        cam = torch.sum(gradients.unsqueeze(0) * activations, dim=1)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        # Resize to original image size
        cam_np = cam.cpu().detach().numpy()
        cam_np = cv2.resize(cam_np, (224, 224))

        return cam_np

    def generate_damage_heatmap(
        self,
        image_path: str,
        output_path: str | None = None,
    ) -> Tuple[str, np.ndarray]:
        """Generate and visualize damage heatmap.

        Args:
            image_path: Path to input image.
            output_path: Path to save heatmap overlay. If None, saves in same dir.

        Returns:
            Tuple of (output_path, heatmap).
        """
        # Generate CAM
        cam = self.generate_cam(image_path, target_class=1)

        # Load original image
        image = Image.open(image_path).convert("RGB")
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_np = cv2.resize(image_np, (224, 224))

        # Apply colormap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        # Save
        if output_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            output_dir = Path(os.getenv("FLEET_HEATMAP_DIR", repo_root / "outputs" / "heatmaps"))
            output_dir.mkdir(parents=True, exist_ok=True)

            img_path = Path(image_path)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            output_path = str(output_dir / f"{img_path.stem}_{timestamp}_heatmap.jpg")

        cv2.imwrite(output_path, overlay)
        return output_path, overlay


def generate_damage_heatmap(
    image_path: str,
    model_path: str,
    device_type: str = "auto",
) -> Tuple[str, np.ndarray]:
    """Generate damage heatmap for an image.

    Args:
        image_path: Path to car image.
        model_path: Path to trained model.
        device_type: Device type ('cuda', 'cpu', or 'auto').

    Returns:
        Tuple of (output_image_path, heatmap_array).
    """
    from .inference import load_trained_model

    device = torch.device("cuda" if (device_type == "auto" and torch.cuda.is_available()) else device_type)
    model, _ = load_trained_model(Path(model_path), device)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    grad_cam = GradCAMViT(model, processor, device)
    return grad_cam.generate_damage_heatmap(image_path)
