"""Model A wrapper for fleet tier identification."""
from __future__ import annotations

from models.model_a.inference import classify_car_type

TIER_MAPPING = {
    "Sedan": "UberX",
    "SUV": "UberXL",
    "VAN": "UberXL",
    "Hatchback": "UberX",
    "Coupe": "UberX",
    "Convertible": "Uber Black",
    "Pick-Up": "UberXL",
}


def identify_vehicle_tier(front_image_path: str, model_path: str = "weights/model a/best_body_classifier.pth") -> dict:
    """Classify car body type and map it to a dispatch tier."""
    body_type, confidence = classify_car_type(front_image_path, model_path=model_path)
    tier = TIER_MAPPING.get(body_type, "UberX")
    return {
        "body_type": body_type,
        "body_confidence": round(confidence, 4),
        "uber_tier": tier,
    }
