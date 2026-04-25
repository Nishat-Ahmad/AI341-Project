"""Multi-view async inspection logic."""
import asyncio
from pathlib import Path
from typing import List, Tuple

from .grad_cam import generate_damage_heatmap
from .inference import classify_damage


async def inspect_single_angle(
    image_path: str,
    angle_name: str,
    model_path: str,
) -> Tuple[str, str, float, str | None]:
    """Inspect a single angle (async).

    Args:
        image_path: Path to angle image.
        angle_name: Angle label (Front, Back, etc).
        model_path: Path to trained model.

    Returns:
        Tuple of (angle, class, confidence, heatmap_path).
    """
    try:
        class_name, confidence = classify_damage(image_path, model_path, device="cpu")

        heatmap_path = None
        if class_name == "Damaged":
            heatmap_path, _ = generate_damage_heatmap(image_path, model_path, device_type="cpu")

        return angle_name, class_name, confidence, heatmap_path
    except Exception as e:
        print(f"Error inspecting {angle_name}: {e}")
        return angle_name, "ERROR", 0.0, None


async def inspect_all_angles(
    image_paths: List[str],
    model_path: str = "weights/model b/best_damage_detector.pth",
    damage_threshold: float = 0.85,
) -> Tuple[str, List[dict]]:
    """Inspect vehicle from all angles.

    Expects images in order: [Front, Back, Left, Right, Roof]

    Args:
        image_paths: List of 5 image paths.
        model_path: Path to trained model.
        damage_threshold: Confidence threshold for rejection.

    Returns:
        Tuple of (verdict, results_list).
        Verdict: 'CLEARED' or 'REJECTED'
        Results: List of dicts with angle, class, confidence, heatmap_path
    """
    if len(image_paths) != 5:
        raise ValueError("Expected exactly 5 images (Front, Back, Left, Right, Roof)")

    angle_names = ["Front", "Back", "Left", "Right", "Roof"]

    # Run all inspections in parallel
    tasks = [
        inspect_single_angle(image_paths[i], angle_names[i], model_path)
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)

    # Process results
    inspection_results = []
    verdict = "CLEARED"

    for angle, class_name, confidence, heatmap_path in results:
        result_dict = {
            "angle": angle,
            "status": class_name,
            "confidence": confidence,
            "heatmap": heatmap_path,
        }
        inspection_results.append(result_dict)

        # Rejection logic: ANY angle with Damaged > threshold = REJECT
        if class_name == "Damaged" and confidence > damage_threshold:
            verdict = "REJECTED"

    return verdict, inspection_results


def inspect_vehicle_sync(
    image_paths: List[str],
    model_path: str = "weights/model b/best_damage_detector.pth",
    damage_threshold: float = 0.85,
) -> Tuple[str, List[dict]]:
    """Synchronous wrapper for vehicle inspection.

    Args:
        image_paths: List of 5 image paths.
        model_path: Path to trained model.
        damage_threshold: Confidence threshold for rejection.

    Returns:
        Tuple of (verdict, results_list).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            inspect_all_angles(image_paths, model_path, damage_threshold)
        )
    finally:
        loop.close()
