"""
A processor aka dataloader for bufr files
> Drafted #177 PR - focusing around ADPUPA as of now
"""

from typing import List, Optional

from eccodes import (
    codes_handle_new_from_file,
    codes_release,
    codes_set,
)

from .schema_for_bufr.adpupa import adpupa_obs
from .schema_for_bufr.base import GeoPoint


class BUFR_Process:
    """
    Low level bufr file decoder
    """

    def __init__(self, schema):
        """
        Schema : Majorly focusing around ADPUPA as of now
        """
        supported = {"adpupa"}
        if schema not in supported:
            raise ValueError(f"Unsupported schema '{schema}'. Supported : {supported}")
        self.schema = schema

    def decode_file(self, path: str) -> List[adpupa_obs]:
        """
        Decode an entire BUFR file into ADPUPA dataclasses.
        Returns a list (because a file may contain multiple soundings).
        """
        observations = []

        with open(path, "rb") as fh:
            while True:
                h = codes_handle_new_from_file(fh, "BUFR")
                if not h:
                    break

                try:
                    codes_set(h, "unpack", 1)
                except Exception:
                    codes_release(h)
                    continue

                obs = self._decode_adpupa(h, file_path=path)
                if obs is not None:
                    observations.append(obs)

                codes_release(h)

        return observations

    def _decode_adpupa(self, h, file_path: str) -> Optional[adpupa_obs]:
        """
        Decode one BUFR message for adpupa
        returns adpupa obs or none
        """
        station_id = self._safe_str(h, "stationIdentifier")

        year = self._safe(h, "year")
        month = self._safe(h, "month")
        day = self._safe(h, "day")
        hour = self._safe(h, "hour")
        minute = self._safe(h, "minute")

        if not all([year, month, day, hour, minute]):
            return None

        lat = self._safe(h, "latitude")
        lon = self._safe(h, "longitude")
        elev = self._safe(h, "heightOfStation")

        location = GeoPoint(lat=lat, lon=lon, elevation_m=elev)

        report_type = self._safe(h, "reportType")
        subtype = self._safe(h, "dataSubCategory")
        inst_type = self._safe(h, "instrumentType")
        balloon_type = self._safe(h, "balloonOrSolarRadiation")
        wind_method = self._safe(h, "methodOfWindMeasurement")

        #  Decode levels
        mandatory = self._decode_level_sequence(h, "mandatory")
        sig_temp = self._decode_level_sequence(h, "significantTemperature")
        sig_wind = self._decode_level_sequence(h, "significantWind")
        trop = self._decode_level_sequence(h, "tropopause")
        maxwind = self._decode_level_sequence(h, "maximumWind")
        datetime_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
        obs = adpupa_obs(
            station_id=station_id,
            datetime=datetime_str,
            location=location,
            report_type=report_type,
            data_subcategory=subtype,
            instrument_type=inst_type,
            balloon_type=balloon_type,
            wind_method=wind_method,
            mandatory_levels=mandatory,
            significant_temperature_levels=sig_temp,
            significant_wind_levels=sig_wind,
            tropopause_levels=trop,
            max_wind_levels=maxwind,
            file_source=file_path,
            bufr_message_index=self._safe(h, "bufrHeaderCentre"),
        )

        return obs

    def _decode_level_sequence(self, h, kind):
        """
        Extracts one of the ADPUPA sequences.
        """
        if kind == "mandatory":
            key_map = {
                "pressure": ["airPressure", "pressure"],
                "height": ["height"],
                "temperature": ["airTemperature"],
                "dewpoint": ["dewpointTemperature"],
                "wind_dir": ["windDirection"],
                "wind_speed": ["windSpeed"],
                "significance": ["significance"],
            }
        elif kind == "significantTemperature":
            key_map = {
                "pressure": ["airPressure", "pressure"],
                "height": ["height"],  # often absent
                "temperature": ["airTemperature"],
                "dewpoint": ["dewpointTemperature"],
                "wind_dir": [],
                "wind_speed": [],
                "significance": ["significance"],
            }
        elif kind == "significantWind":
            key_map = {
                "pressure": ["airPressure", "pressure"],
                "height": [],
                "temperature": [],
                "dewpoint": [],
                "wind_dir": ["windDirection"],
                "wind_speed": ["windSpeed"],
                "significance": ["significance"],
            }
        else:
            # tropopause / maxwind have mixed keys
            key_map = {
                "pressure": ["airPressure", "pressure"],
                "height": ["height"],
                "temperature": ["airTemperature"],
                "dewpoint": ["dewpointTemperature"],
                "wind_dir": ["windDirection"],
                "wind_speed": ["windSpeed"],
                "significance": ["significance"],
            }

        # Extract arrays
        arrays = {
            k: self._first_present_array(h, arr_keys)
            for k, arr_keys in key_map.items()
        }

        # Determine length
        length = 0
        for arr in arrays.values():
            if isinstance(arr, list) and len(arr) > length:
                length = len(arr)

        levels = []
        for i in range(length):
            level = adpupa_level(
                pressure_hPa=self._index_or_none(arrays["pressure"], i),
                geopotential_height_m=self._index_or_none(arrays["height"], i),
                temperature_K=self._index_or_none(arrays["temperature"], i),
                dewpoint_K=self._index_or_none(arrays["dewpoint"], i),
                wind_direction_deg=self._index_or_none(arrays["wind_dir"], i),
                wind_speed_ms=self._index_or_none(arrays["wind_speed"], i),
                significance=self._index_or_none(arrays["significance"], i),
            )
            levels.append(level)

        return levels
    def _safe(self, h, key: str) -> Any:
        try:
            return codes_get(h, key)
        except Exception:
            return None

    def _safe_str(self, h, key: str) -> Optional[str]:
        val = self._safe(h, key)
        return str(val) if val is not None else None

    def _safe_array(self, h, key: str) -> Optional[List[Any]]:
        try:
            return list(codes_get_array(h, key))
        except Exception:
            return None

    def _first_present_array(self, h, keys: List[str]) -> Optional[List[Any]]:
        for k in keys:
            arr = self._safe_array(h, k)
            if arr is not None:
                return arr
        return None

    def _index_or_none(self, arr: Optional[List[Any]], i: int) -> Any:
        if arr is None or i >= len(arr):
            return None
        return arr[i]
