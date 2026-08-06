from dataclasses import dataclass
from typing import Optional

from .base import GeoPoint


@dataclass
class adpupa_level:
    pressure_hPa: Optional[float]
    geopotential_height_m: Optional[float]
    temperature_K: Optional[float]
    dewpoint_K: Optional[float]
    wind_direction_deg: Optional[float]
    wind_speed_ms: Optional[float]
    significance: Optional[int]

    qc_pressure: Optional[int] = None
    qc_height: Optional[int] = None
    qc_temperature: Optional[int] = None
    qc_dewpoint: Optional[int] = None
    qc_wind_dir: Optional[int] = None
    qc_wind_speed: Optional[int] = None


@dataclass
class adpupa_obs:
    # metadata
    station_id: str
    datetime: str  # or datetime.datetime
    location: GeoPoint

    # radiosonde metadata
    report_type: Optional[int] = None
    data_subcategory: Optional[int] = None
    instrument_type: Optional[int] = None
    balloon_type: Optional[int] = None
    wind_method: Optional[int] = None

    # levels
    mandatory_levels: List[adpupa_level] = None
    significant_temperature_levels: List[adpupa_level] = None
    significant_wind_levels: List[adpupa_level] = None
    tropopause_levels: List[adpupa_level] = None
    max_wind_levels: List[adpupa_level] = None

    # provenance
    file_source: Optional[str] = None
    bufr_message_index: Optional[int] = None
