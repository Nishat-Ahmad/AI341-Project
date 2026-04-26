"""Routing utility powered by OpenRouteService (ORS)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import openrouteservice


class RoutingServiceError(RuntimeError):
    """Raised when ORS routing or geocoding fails."""


@dataclass(frozen=True)
class RouteSummary:
    distance_km: float
    duration_mins: float
    formatted_eta: str


def _load_env_file() -> None:
    """Lightweight .env loader to avoid hard dependency on dotenv package."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_ors_client() -> openrouteservice.Client:
    _load_env_file()
    api_key = os.getenv("ORS_API_KEY", "").strip()
    if not api_key:
        raise RoutingServiceError("Missing ORS_API_KEY. Set it in .env or environment variables.")
    return openrouteservice.Client(key=api_key)


def _format_eta(duration_minutes: float) -> str:
    total_mins = int(round(duration_minutes))
    hours, mins = divmod(total_mins, 60)
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def geocode_destination(destination: str) -> Tuple[float, float]:
    """Convert a destination string into ORS coordinates (lon, lat)."""
    if not destination or not destination.strip():
        raise RoutingServiceError("Destination text is empty.")

    client = _get_ors_client()
    try:
        result = client.pelias_search(text=destination.strip(), size=1)
        features = result.get("features", [])
        if not features:
            raise RoutingServiceError(f"No geocoding result for destination: {destination}")
        coords = features[0]["geometry"]["coordinates"]
        return float(coords[0]), float(coords[1])
    except RoutingServiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RoutingServiceError(f"Geocoding failed: {exc}") from exc


def get_ride_details(start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> dict:
    """Get driving distance and ETA via ORS `driving-car` profile.

    Args:
        start_coords: Tuple(lon, lat) for origin.
        end_coords: Tuple(lon, lat) for destination.

    Returns:
        Dictionary with distance_km, duration_mins, formatted_eta.
    """
    client = _get_ors_client()

    try:
        route = client.directions(
            coordinates=[list(start_coords), list(end_coords)],
            profile="driving-car",
            format="geojson",
        )
        segment = route["features"][0]["properties"]["segments"][0]
        distance_km = float(segment["distance"]) / 1000.0
        duration_mins = float(segment["duration"]) / 60.0

        summary = RouteSummary(
            distance_km=round(distance_km, 2),
            duration_mins=round(duration_mins, 2),
            formatted_eta=_format_eta(duration_mins),
        )
        return {
            "distance_km": summary.distance_km,
            "duration_mins": summary.duration_mins,
            "formatted_eta": summary.formatted_eta,
        }
    except Exception as exc:  # noqa: BLE001
        raise RoutingServiceError(f"Routing request failed: {exc}") from exc
