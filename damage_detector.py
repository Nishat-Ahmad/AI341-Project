"""Model B wrapper for multi-angle damage inspection."""
from __future__ import annotations

import asyncio
from typing import Dict, List

from models.model_b.grad_cam import generate_damage_heatmap
from models.model_b.inference import classify_damage

ANGLE_ORDER = ["Front", "Back", "Left", "Right", "Roof"]


async def inspect_angle(angle: str, image_path: str, model_path: str) -> dict:
    """Classify a single image and generate Grad-CAM if damaged."""

    def _run_sync() -> dict:
        status, confidence = classify_damage(image_path, model_path=model_path)
        heatmap_path = None
        if status == "Damaged":
            heatmap_path, _ = generate_damage_heatmap(
                image_path=image_path,
                model_path=model_path,
                device_type="cpu",
            )
        return {
            "angle": angle,
            "status": status,
            "confidence": round(confidence, 4),
            "heatmap_path": heatmap_path,
        }

    return await asyncio.to_thread(_run_sync)


async def inspect_all_angles(image_paths: Dict[str, str], model_path: str = "weights/model b/best_damage_detector.pth") -> List[dict]:
    """Run Model B in parallel over 5 required angles."""
    missing = [name for name in ANGLE_ORDER if name not in image_paths]
    if missing:
        raise ValueError(f"Missing angles: {missing}")

    tasks = [inspect_angle(angle, image_paths[angle], model_path) for angle in ANGLE_ORDER]
    results = await asyncio.gather(*tasks)
    return results
