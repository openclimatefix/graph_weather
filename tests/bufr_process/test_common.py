import pandas as pd
from graph_weather.data.bufr_process import FieldMapping, ADPUPA_schema, CRIS_schema
from typing import Any


class MockADPUPASchema(ADPUPA_schema):
    """Mock ADPUPA schema with proper FieldMapping dtypes."""

    def _build_mappings(self):
        self.field_mappings = {
            "latitude": FieldMapping(
                source_name="latitude",
                output_name="LAT",
                dtype=float,
                description="Station latitude",
            ),
            "longitude": FieldMapping(
                source_name="longitude",
                output_name="LON",
                dtype=float,
                description="Station longitude",
            ),
            "obsTime": FieldMapping(
                source_name="obsTime",
                output_name="OBS_TIMESTAMP",
                dtype=object,
                transform_fn=self._convert_timestamp,
                description="Observation timestamp",
            ),
            "airTemperature": FieldMapping(
                source_name="airTemperature",
                output_name="temperature",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                description="Temperature in Celsius",
            ),
            "pressure": FieldMapping(
                source_name="pressure",
                output_name="pressure",
                dtype=float,
                description="Pressure in Pa",
            ),
            "height": FieldMapping(
                source_name="height",
                output_name="height",
                dtype=float,
                description="Height above sealevel in m",
            ),
            "dewpointTemperature": FieldMapping(
                source_name="dewpointTemperature",
                output_name="dew_point",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                description="Dew point in Celsius",
            ),
            "windU": FieldMapping(
                source_name="windU",
                output_name="u_wind",
                dtype=float,
                description="U-component wind (m/s)",
            ),
            "windV": FieldMapping(
                source_name="windV",
                output_name="v_wind",
                dtype=float,
                description="V-component wind (m/s)",
            ),
            "stationId": FieldMapping(
                source_name="stationId",
                output_name="station_id",
                dtype=str,
                required=False,
                description="Station identifier",
            ),
        }

    def _convert_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert BUFR timestamp to pandas Timestamp."""
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit="s")
        elif isinstance(value, str):
            return pd.Timestamp(value)
        else:
            return pd.Timestamp(value)


class MockCRISSchema(CRIS_schema):
    """Mock CrIS schema with proper FieldMapping dtypes."""

    def _build_mappings(self):
        self.field_mappings = {
            "latitude": FieldMapping(
                source_name="latitude",
                output_name="LAT",
                dtype=float,
                description="Satellite latitude",
            ),
            "longitude": FieldMapping(
                source_name="longitude",
                output_name="LON",
                dtype=float,
                description="Satellite longitude",
            ),
            "obsTime": FieldMapping(
                source_name="obsTime",
                output_name="OBS_TIMESTAMP",
                dtype=object,
                transform_fn=self._convert_timestamp,
                description="Observation timestamp",
            ),
            "retrievedTemperature": FieldMapping(
                source_name="retrievedTemperature",
                output_name="temperature",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                description="Retrieved temperature in Celsius",
            ),
            "retrievedPressure": FieldMapping(
                source_name="retrievedPressure",
                output_name="pressure",
                dtype=float,
                description="Retrieved pressure in Pa",
            ),
            "sensorZenithAngle": FieldMapping(
                source_name="sensorZenithAngle",
                output_name="sensor_zenith_angle",
                dtype=float,
                required=False,
                description="Sensor zenith angle",
            ),
            "qualityFlags": FieldMapping(
                source_name="qualityFlags",
                output_name="qc_flag",
                dtype=int,
                description="Quality control flags",
            ),
        }

    def _convert_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert BUFR timestamp to pandas Timestamp."""
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit="s")
        elif isinstance(value, str):
            return pd.Timestamp(value)
        else:
            return pd.Timestamp(value)
