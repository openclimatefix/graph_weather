"""Shared utility functions for graph weather models."""

from typing import Sequence, Tuple


def validate_lat_lons(lat_lons: Sequence[Tuple[float, float]]) -> None:
    """Validate a non-empty sequence of latitude and longitude pairs."""
    if not lat_lons:
        raise ValueError("lat_lons must not be empty.")
    for index, (lat, _lon) in enumerate(lat_lons):
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Coordinate {index}: latitude {lat} is outside [-90, 90].")
