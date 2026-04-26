"""Fleet-Vision orchestrator brain (LangChain + model workflow)."""
from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Dict

from langchain_core.runnables import RunnableLambda

from body_classifier import identify_vehicle_tier
from damage_detector import inspect_all_angles
from utils.routing_service import geocode_destination, get_ride_details


class FleetVisionOrchestrator:
    """Coordinates tier classification, safety gate, and routing."""

    def __init__(self) -> None:
        base_lon = float(os.getenv("FLEET_BASE_LON", "77.2090"))
        base_lat = float(os.getenv("FLEET_BASE_LAT", "28.6139"))
        self.base_coords = (base_lon, base_lat)

        self._chain = RunnableLambda(self._run_sync)

    def _run_sync(self, payload: dict) -> dict:
        image_paths = payload["image_paths"]
        destination = payload["destination"]
        return asyncio.run(self.run(image_paths=image_paths, destination=destination))

    def chain(self) -> RunnableLambda:
        """Expose LangChain runnable for integration."""
        return self._chain

    async def run(self, image_paths: Dict[str, str], destination: str) -> dict:
        """Execute full workflow and return dispatch report JSON."""
        required = ["Front", "Back", "Left", "Right", "Roof"]
        missing = [k for k in required if k not in image_paths]
        if missing:
            raise ValueError(f"Missing required angles: {missing}")

        # Tier Identification (Model A on Front image)
        tier_info = identify_vehicle_tier(image_paths["Front"])

        # Safety Gate (Model B in parallel for 5 angles)
        inspection_results = await inspect_all_angles(image_paths)
        damaged = [item for item in inspection_results if item["status"] == "Damaged"]

        if damaged:
            return {
                "status": "REJECTED",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "tier_info": tier_info,
                "damaged_angles": [d["angle"] for d in damaged],
                "inspection_results": inspection_results,
                "heatmaps": {
                    d["angle"]: d["heatmap_path"]
                    for d in damaged
                    if d["heatmap_path"]
                },
            }

        # Route only when safety gate passes.
        destination_coords = geocode_destination(destination)
        route_details = get_ride_details(self.base_coords, destination_coords)

        return {
            "status": "APPROVED",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "tier_info": tier_info,
            "destination": destination,
            "origin_coords": {
                "lon": self.base_coords[0],
                "lat": self.base_coords[1],
            },
            "destination_coords": {
                "lon": destination_coords[0],
                "lat": destination_coords[1],
            },
            "route": route_details,
            "inspection_results": inspection_results,
        }
