import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Bufr Processor")

from torch.utils.data import IterableDataset

try:
    import eccodes
except ImportError:
    raise ImportError("eccodes not installed. Install with: `pip install eccodes`")


@dataclass
class FieldMapping:
    """Maps a BUFR source field to NNJA-AI output field."""

    source_name: str
    output_name: str
    dtype: type
    transform_fn: Optional[Callable] = None
    required: bool = True
    description: str = ""

    def apply(self, value: Any) -> Any:
        """Apply transformation to source value."""
        if value is None:
            return None
        if self.transform_fn:
            return self.transform_fn(value)
        return value


class NNJA_Schema:
    """
    Defines the canonical NNJA-AI schema that all BUFR data maps to

    Mimics NNJA-AI's xarray format with standardized coordinates and variables.
    Coordinate system matches NNJA-AI:
    - OBS_TIMESTAMP: observation time (ns precision)
    - LAT: latitude
    - LON: longitude
    """

    COORDINATES = {
        "OBS_TIMESTAMP": "datetime64[ns]",
        "LAT": "float32",
        "LON": "float32",
    }

    VARIABLES = {
        "temperature": "float32",
        "pressure": "float32",
        "relative_humidity": "float32",
        "u_wind": "float32",
        "v_wind": "float32",
        "dew_point": "float32",
        "height": "float32",
    }

    ATTRIBUTES = {
        "source": "DATA_SOURCE",
        "qc_flag": "int8",
        "processing_timestamp": "datetime64[ns]",
    }

    @classmethod
    def to_xarray_schema(cls) -> Dict[str, str]:
        """Get full schema as dict for xarray construction."""
        return {**cls.COORDINATES, **cls.VARIABLES, **cls.ATTRIBUTES}

    @classmethod
    def get_coordinate_names(cls) -> List[str]:
        """Get list of coordinate names."""
        return list(cls.COORDINATES.keys())

    @classmethod
    def validate_data(cls, data: Dict[str, np.ndarray]) -> bool:
        """Check if data has required NNJA coordinates."""
        required_coords = ["OBS_TIMESTAMP", "LAT", "LON"]
        return all(coord in data for coord in required_coords)


class DataSourceSchema:
    """
    Abstract base for source-specific BUFR schema mappings.
    Defines how BUFR fields from a specific source (ADPUPA, CrIS, etc.)
    map to NNJA-AI canonical format.
    """

    source_name: str = "unknown"

    def __init__(self):
        self.field_mappings: Dict[str, FieldMapping] = {}
        self._build_mappings()
        self._validate()

    def _build_mappings(self):
        """
        Override in subclasses to define BUFR → NNJA field mappings.

        Example:
            self.field_mappings['T'] = FieldMapping(
                source_name='T',
                output_name='temperature',
                dtype=float,
                transform_fn=lambda x: x - 273.15,  # K to C
                description='Temperature in Celsius'
            )
        """
        raise NotImplementedError("Subclasses must implement _build_mappings()")

    def _validate(self):
        """Ensure all required NNJA coordinates are mapped."""
        required = ["OBS_TIMESTAMP", "LAT", "LON"]
        mapped_outputs = {m.output_name for m in self.field_mappings.values()}
        missing = [r for r in required if r not in mapped_outputs]
        if missing:
            logger.warning(f"{self.source_name} schema missing required outputs: {missing}")

    def map_observation(self, bufr_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw BUFR message to NNJA-AI format.
        Always include all mapped output fields, even if missing from the BUFR message.
        """
        mapped = {}

        for field_map in self.field_mappings.values():
            if field_map.source_name in bufr_message:
                raw_value = bufr_message[field_map.source_name]
                try:
                    value = field_map.apply(raw_value)
                    mapped[field_map.output_name] = value
                except Exception as e:
                    logger.warning(f"Error transforming {field_map.source_name}: {e}")
                    mapped[field_map.output_name] = None
            else:
                # Field not present — default to None
                mapped[field_map.output_name] = None

        return mapped


class ADPUPA_schema(DataSourceSchema):
    """ADPUPA (upper-air radiosonde) BUFR schema mapping to NNJA-AI.
    
    Includes mandatory pressure levels: 1000, 925, 850, 700, 500, 300, 200, 100 hPa
    """

    source_name = "ADPUPA"
    
    # Standard mandatory pressure levels in Pa
    MANDATORY_LEVELS = [100, 200, 300, 500, 700, 850, 925, 1000]

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
            
            # ===== STATION METADATA =====
            "WMOB": FieldMapping(
                source_name="WMOB",
                output_name="wmo_block_number",
                dtype=str,
                required=False,
                description="WMO block number",
            ),
            "WMOS": FieldMapping(
                source_name="WMOS",
                output_name="wmo_station_number",
                dtype=str,
                required=False,
                description="WMO station number",
            ),
            "WMOR": FieldMapping(
                source_name="WMOR",
                output_name="wmo_region",
                dtype=int,
                required=False,
                description="WMO Region number/geographical area",
            ),
            "UASID.RPID": FieldMapping(
                source_name="UASID.RPID",
                output_name="report_id",
                dtype=str,
                required=False,
                description="Report identifier",
            ),
            "UASID.SELV": FieldMapping(
                source_name="UASID.SELV",
                output_name="station_elevation",
                dtype=float,
                required=False,
                description="Height of station (m)",
            ),
            "stationId": FieldMapping(
                source_name="stationId",
                output_name="station_id",
                dtype=str,
                required=False,
                description="Station identifier",
            ),
            
            # ===== SURFACE/SINGLE LEVEL DATA =====
            "airTemperature": FieldMapping(
                source_name="airTemperature",
                output_name="temperature",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                required=False,
                description="Surface temperature in Celsius",
            ),
            "pressure": FieldMapping(
                source_name="pressure",
                output_name="pressure",
                dtype=float,
                required=False,
                description="Surface pressure in Pa",
            ),
            "height": FieldMapping(
                source_name="height",
                output_name="height",
                dtype=float,
                required=False,
                description="Height above sea level in m",
            ),
            "dewpointTemperature": FieldMapping(
                source_name="dewpointTemperature",
                output_name="dew_point",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                required=False,
                description="Surface dew point in Celsius",
            ),
            "windU": FieldMapping(
                source_name="windU",
                output_name="u_wind",
                dtype=float,
                required=False,
                description="Surface U-component wind (m/s)",
            ),
            "windV": FieldMapping(
                source_name="windV",
                output_name="v_wind",
                dtype=float,
                required=False,
                description="Surface V-component wind (m/s)",
            ),
            
            # ===== ADDITIONAL METEOROLOGICAL DATA =====
            "UASDG.SST1": FieldMapping(
                source_name="UASDG.SST1",
                output_name="sea_surface_temp",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                required=False,
                description="Sea/water temperature in Celsius",
            ),
            "UAADF.STBS5": FieldMapping(
                source_name="UAADF.STBS5",
                output_name="showalter_index",
                dtype=float,
                required=False,
                description="Modified Showalter stability index",
            ),
            "UAADF.MWDL": FieldMapping(
                source_name="UAADF.MWDL",
                output_name="mean_wind_dir_low",
                dtype=float,
                required=False,
                description="Mean wind direction for surface - 1500m (degrees)",
            ),
            "UAADF.MWSL": FieldMapping(
                source_name="UAADF.MWSL",
                output_name="mean_wind_speed_low",
                dtype=float,
                required=False,
                description="Mean wind speed for surface - 1500m (m/s)",
            ),
            "UAADF.MWDH": FieldMapping(
                source_name="UAADF.MWDH",
                output_name="mean_wind_dir_high",
                dtype=float,
                required=False,
                description="Mean wind direction for 1500-3000m (degrees)",
            ),
            "UAADF.MWSH": FieldMapping(
                source_name="UAADF.MWSH",
                output_name="mean_wind_speed_high",
                dtype=float,
                required=False,
                description="Mean wind speed for 1500-3000m (m/s)",
            ),
            
            "MSG_TYPE": FieldMapping(
                source_name="MSG_TYPE",
                output_name="message_type",
                dtype=str,
                required=False,
                description="Source message type",
            ),
            "MSG_DATE": FieldMapping(
                source_name="MSG_DATE",
                output_name="message_date",
                dtype=object,
                transform_fn=self._convert_timestamp,
                required=False,
                description="Message valid timestamp",
            ),
            "OBS_DATE": FieldMapping(
                source_name="OBS_DATE",
                output_name="obs_date",
                dtype=object,
                transform_fn=self._convert_timestamp,
                required=False,
                description="Date of the observation",
            ),
            "SRC_FILENAME": FieldMapping(
                source_name="SRC_FILENAME",
                output_name="source_filename",
                dtype=str,
                required=False,
                description="Source filename",
            ),
        }
        
        for level_hpa in self.MANDATORY_LEVELS:
            level_pa = level_hpa * 100  # Convert hPa to Pa for BUFR field names
            
            self.field_mappings[f"TMDB_PRLC{level_pa}"] = FieldMapping(
                source_name=f"TMDB_PRLC{level_pa}",
                output_name=f"temperature_{level_hpa}hPa",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                required=False,
                description=f"Temperature at {level_hpa} hPa in Celsius",
            )
            
            # Dewpoint at pressure level
            self.field_mappings[f"TMDP_PRLC{level_pa}"] = FieldMapping(
                source_name=f"TMDP_PRLC{level_pa}",
                output_name=f"dew_point_{level_hpa}hPa",
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                required=False,
                description=f"Dewpoint temperature at {level_hpa} hPa in Celsius",
            )
            
            # Wind speed at pressure level
            self.field_mappings[f"WSPD_PRLC{level_pa}"] = FieldMapping(
                source_name=f"WSPD_PRLC{level_pa}",
                output_name=f"wind_speed_{level_hpa}hPa",
                dtype=float,
                required=False,
                description=f"Wind speed at {level_hpa} hPa (m/s)",
            )
            
            # Wind direction at pressure level
            self.field_mappings[f"WDIR_PRLC{level_pa}"] = FieldMapping(
                source_name=f"WDIR_PRLC{level_pa}",
                output_name=f"wind_direction_{level_hpa}hPa",
                dtype=float,
                required=False,
                description=f"Wind direction at {level_hpa} hPa (degrees)",
            )
            
            # Geopotential at pressure level
            self.field_mappings[f"GP10_PRLC{level_pa}"] = FieldMapping(
                source_name=f"GP10_PRLC{level_pa}",
                output_name=f"geopotential_{level_hpa}hPa",
                dtype=float,
                required=False,
                description=f"Geopotential at {level_hpa} hPa (m²/s²)",
            )
            
            # ===== QUALITY CONTROL FLAGS =====
            self.field_mappings[f"QMAT_PRLC{level_pa}"] = FieldMapping(
                source_name=f"QMAT_PRLC{level_pa}",
                output_name=f"qc_temperature_{level_hpa}hPa",
                dtype=int,
                required=False,
                description=f"QC flag for temperature at {level_hpa} hPa",
            )
            
            self.field_mappings[f"QMDD_PRLC{level_pa}"] = FieldMapping(
                source_name=f"QMDD_PRLC{level_pa}",
                output_name=f"qc_moisture_{level_hpa}hPa",
                dtype=int,
                required=False,
                description=f"QC flag for moisture at {level_hpa} hPa",
            )
            
            self.field_mappings[f"QMWN_PRLC{level_pa}"] = FieldMapping(
                source_name=f"QMWN_PRLC{level_pa}",
                output_name=f"qc_wind_{level_hpa}hPa",
                dtype=int,
                required=False,
                description=f"QC flag for wind at {level_hpa} hPa",
            )
            
            self.field_mappings[f"QMGP_PRLC{level_pa}"] = FieldMapping(
                source_name=f"QMGP_PRLC{level_pa}",
                output_name=f"qc_geopotential_{level_hpa}hPa",
                dtype=int,
                required=False,
                description=f"QC flag for geopotential at {level_hpa} hPa",
            )
            
            self.field_mappings[f"QMPR_PRLC{level_pa}"] = FieldMapping(
                source_name=f"QMPR_PRLC{level_pa}",
                output_name=f"qc_pressure_{level_hpa}hPa",
                dtype=int,
                required=False,
                description=f"QC flag for pressure at {level_hpa} hPa",
            )
            
            # Vertical sounding significance
            self.field_mappings[f"VSIG_PRLC{level_pa}"] = FieldMapping(
                source_name=f"VSIG_PRLC{level_pa}",
                output_name=f"sounding_significance_{level_hpa}hPa",
                dtype=int,
                required=False,
                description=f"Vertical sounding significance at {level_hpa} hPa",
            )

    def _convert_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert BUFR timestamp to pandas Timestamp."""
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit="s")
        elif isinstance(value, str):
            return pd.Timestamp(value)
        else:
            return pd.Timestamp(value)

class CRIS_schema(DataSourceSchema):
    """CrIS (satellite hyperspectral) BUFR schema mapping to NNJA-AI."""

    source_name = "CrIS"

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
            "obsDate": FieldMapping(
                source_name="obsDate",
                output_name="OBS_DATE",
                dtype=object,
                description="Date of the observation",
            ),
            "satelliteId": FieldMapping(
                source_name="satelliteId",
                output_name="SAID",
                dtype=int,
                description="Satellite identifier",
            ),
            "sensorZenithAngle": FieldMapping(
                source_name="sensorZenithAngle",
                output_name="SAZA",
                dtype=float,
                required=False,
                description="Satellite zenith angle",
            ),
            "solarZenithAngle": FieldMapping(
                source_name="solarZenithAngle",
                output_name="SOZA",
                dtype=float,
                required=False,
                description="Solar zenith angle",
            ),
            "solarAzimuth": FieldMapping(
                source_name="solarAzimuth",
                output_name="SOLAZI",
                dtype=float,
                required=False,
                description="Solar azimuth angle",
            ),
            "bearingAzimuth": FieldMapping(
                source_name="bearingAzimuth",
                output_name="BEARAZ",
                dtype=float,
                required=False,
                description="Bearing or azimuth",
            ),
            "orbitNumber": FieldMapping(
                source_name="orbitNumber",
                output_name="ORBN",
                dtype=int,
                required=False,
                description="Orbit number",
            ),
            "scanLineNumber": FieldMapping(
                source_name="scanLineNumber",
                output_name="SLNM",
                dtype=int,
                required=False,
                description="Scan line number",
            ),
            "fieldOfRegardNumber": FieldMapping(
                source_name="fieldOfRegardNumber",
                output_name="FORN",
                dtype=int,
                required=False,
                description="Field of regard number",
            ),
            "fieldOfViewNumber": FieldMapping(
                source_name="fieldOfViewNumber",
                output_name="FOVN",
                dtype=int,
                required=False,
                description="Field of view number",
            ),
            "heightAboveSurface": FieldMapping(
                source_name="heightAboveSurface",
                output_name="HMSL",
                dtype=float,
                required=False,
                description="Height or altitude above mean sea level",
            ),
            "heightOfLandSurface": FieldMapping(
                source_name="heightOfLandSurface",
                output_name="HOLS",
                dtype=float,
                required=False,
                description="Height of land surface",
            ),
            "totalCloudCover": FieldMapping(
                source_name="totalCloudCover",
                output_name="TOCC",
                dtype=float,
                required=False,
                description="Cloud cover (total)",
            ),
            "cloudTopHeight": FieldMapping(
                source_name="cloudTopHeight",
                output_name="HOCT",
                dtype=float,
                required=False,
                description="Height of top of cloud",
            ),
            "landFraction": FieldMapping(
                source_name="landFraction",
                output_name="ALFR",
                dtype=float,
                required=False,
                description="Land fraction",
            ),
            "landSeaQualifier": FieldMapping(
                source_name="landSeaQualifier",
                output_name="LSQL",
                dtype=int,
                required=False,
                description="Land/sea qualifier",
            ),
            "qualityFlags": FieldMapping(
                source_name="qualityFlags",
                output_name="NSQF",
                dtype=int,
                required=False,
                description="Scan-level quality flags",
            ),
            "radianceTypeFlags": FieldMapping(
                source_name="radianceTypeFlags",
                output_name="RDTF",
                dtype=int,
                required=False,
                description="Radiance type flags",
            ),
            "geolocationQuality": FieldMapping(
                source_name="geolocationQuality",
                output_name="NGQI",
                dtype=int,
                required=False,
                description="Geolocation quality",
            ),
            "orbitQualifier": FieldMapping(
                source_name="orbitQualifier",
                output_name="STKO",
                dtype=int,
                required=False,
                description="Ascending/descending orbit qualifier",
            ),
            # Channel radiance mappings - you can add specific channels as needed
            "channelRadiances": FieldMapping(
                source_name="channelRadiances",
                output_name="CRCHNM_SRAD01",
                dtype=object,
                required=False,
                description="CrIS channel radiances array",
            ),
            "guardChannelData": FieldMapping(
                source_name="guardChannelData",
                output_name="GCRCHN",
                dtype=object,
                required=False,
                description="NPP CrIS GUARD CHANNEL DATA array",
            ),
            "viirsSceneData": FieldMapping(
                source_name="viirsSceneData",
                output_name="CRISCS",
                dtype=object,
                required=False,
                description="CrIS LEVEL 1B VIIRS SINGLE SCENE SEQUENCE DATA array",
            ),
        }

        # common channels for use case
        common_channels = {
            "radiance_ch19": "CRCHNM.SRAD01_00019",
            "radiance_ch24": "CRCHNM.SRAD01_00024", 
            "radiance_ch26": "CRCHNM.SRAD01_00026",
            "radiance_ch27": "CRCHNM.SRAD01_00027",
            "radiance_ch31": "CRCHNM.SRAD01_00031",
            "radiance_ch32": "CRCHNM.SRAD01_00032",
        }
        
        for key, source_name in common_channels.items():
            self.field_mappings[key] = FieldMapping(
                source_name=source_name,
                output_name=source_name.replace(".", "_"),
                dtype=float,
                required=False,
                description=f"Channel radiance for {source_name}",
            )

    def _convert_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert BUFR timestamp to pandas Timestamp."""
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit="s")
        elif isinstance(value, str):
            return pd.Timestamp(value)
        else:
            return pd.Timestamp(value)
        
class BUFR_processor:
    """
    Low-level BUFR file decoder.
    Handles binary BUFR format decoding using eccodes library.
    """

    def __init__(self, schema: DataSourceSchema):
        """
        Args:
            -> schema : DataSourceSchema instance
        """
        if not isinstance(schema, DataSourceSchema):
            raise TypeError("schema must be of DataSourceSchema instance")

        self.schema = schema

    def decoder_bufr_files(self, filepath) -> List[Dict[str, any]]:
        """Decode all messages from BUFR file."""
        msgs = []
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"BUFR file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                while True:
                    bufr_id = eccodes.codes_bufr_new_from_file(f)
                    if bufr_id is None:
                        break

                    try:
                        eccodes.codes_set(bufr_id, "unpack", 1)
                        msg = {}
                        iterator = eccodes.codes_bufr_keys_iterator_new(bufr_id)
                        while eccodes.codes_bufr_keys_iterator_next(iterator):
                            key = eccodes.codes_bufr_keys_iterator_get_name(iterator)
                            try:
                                value = eccodes.codes_get_string(bufr_id, key)
                                msg[key] = value
                            except Exception:
                                try:
                                    value = eccodes.codes_get_double(bufr_id, key)
                                    msg[key] = value
                                except Exception:
                                    pass
                        eccodes.codes_bufr_keys_iterator_delete(iterator)
                        msgs.append(msg)
                    finally:
                        eccodes.codes_release(bufr_id)
        except Exception as e:
            logger.error(f"Error decoding BUFR file {filepath}: {e}")
            raise

        logger.info(f"Decoded {len(msgs)} messages from {filepath}")
        return msgs

    def process_files_to_dataframe(self, filepath: str) -> pd.DataFrame:
        """
        Decode BUFR file and map to NNJA schema, return as DataFrame.

        Args:
            -> filepath: Path to BUFR file

        Returns:
            -> pandas DataFrame in NNJA-AI format
        """

        raw_msgs = self.decoder_bufr_files(filepath=filepath)

        transformed = []

        for msg in raw_msgs:
            mapped = self.schema.map_observation(msg)
            if mapped:
                transformed.append(mapped)

        if not transformed:
            logger.warning(f"No valid observations found in {filepath}")
            return pd.DataFrame()

        df = pd.DataFrame(transformed)

        for col in df.columns:
            if col in NNJA_Schema.COORDINATES:
                dtype = NNJA_Schema.COORDINATES[col]
                if "datetime" in dtype:
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype.split("[")[0] if "[" in dtype else dtype)
        if not NNJA_Schema.validate_data(df.to_dict(orient="list")):
            logger.warning(f"DataFrame missing required NNJA coordinates from {filepath}")

        return df

    def process_files_to_xarray(self, filepath: str) -> xr.Dataset:
        """
        Process BUFR file to xarray Dataset in NNJA-AI format.

        Args:
            -> filepath: Path to BUFR file

        Returns:
            -> xarray Dataset in NNJA-AI format
        """
        df = self.process_files_to_dataframe(filepath=filepath)

        if df.empty:
            logger.warning(f"No data to convert to xarray from {filepath}")
            return xr.Dataset()

        data_vars = {}
        for col in df.columns:
            if col not in ["OBS_TIMESTAMP", "LAT", "LON"]:
                data_vars[col] = (["observation"], df[col].values)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "obs": df.index,
                "time": ("obs", df["OBS_TIMESTAMP"].values),
                "lat": ("obs", df["LAT"].values),
                "lon": ("obs", df["LON"].values),
            },
        )
        ds.attrs["source"] = self.schema.source_name
        ds.attrs["processing_timestamp"] = pd.Timestamp.now().isoformat()
        ds.attrs["num_observations"] = len(df)

        return ds

    def process_files_to_parquet(self, filepath: str, output_path: str) -> None:
        """
        Process BUFR file and save as Parquet in NNJA-AI format.

        Args:
            -> filepath: Path to BUFR file
            -> output_path: Path for output Parquet file
        """
        df = self.process_files_to_dataframe(filepath=filepath)

        if not df.empty:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(df)} observations to {output_path}")
        else:
            logger.warning(f"No data to save for {filepath}")


class BUFR_dataloader:

    SCHEMA_REGISTRY = {
        "ADPUPA": ADPUPA_schema,
        "CrIS": CRIS_schema,
    }

    def __init__(self, filepath: str, schema_name: Optional[str] = None):
        """
        Args:
            filepath: Path to BUFR file or directory
            schema_name: Data source name ('ADPUPA', 'CrIS', etc.)
        """
        self.filepath = Path(filepath)
        self.schema_name = schema_name or self._infer_schema_from_path()

        if self.schema_name not in self.SCHEMA_REGISTRY:
            raise ValueError(
                f'Unknown schema "{self.schema_name}". Available: {list(self.SCHEMA_REGISTRY.keys())}'
            )

        self.schema = self.SCHEMA_REGISTRY[self.schema_name]()
        self.processor = BUFR_processor(self.schema)

    def _infer_schema_from_path(self) -> str:
        """Infer schema from filename or path patterns."""
        filename = self.filepath.name.lower()

        if "adpupa" in filename or "raob" in filename or "sound" in filename:
            return "ADPUPA"
        elif "cris" in filename:
            return "CrIS"
        elif "iasi" in filename:
            return "IASI"
        elif "atms" in filename:
            return "ATMS"
        else:
            # Default to ADPUPA for now
            logger.warning(f"Could not infer schema from {filename}, defaulting to ADPUPA")
            return "ADPUPA"

    def to_dataframe(self) -> pd.DataFrame:
        """Process BUFR file to DataFrame."""
        return self.processor.process_files_to_dataframe(str(self.filepath))

    def to_xarray(self) -> xr.Dataset:
        """Process BUFR file to xarray Dataset."""
        return self.processor.process_files_to_xarray(str(self.filepath))

    def to_parquet(self, output_path: str) -> None:
        """Process BUFR file to Parquet format."""
        self.processor.process_files_to_parquet(str(self.filepath), output_path)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over observations in the BUFR file."""
        messages = self.processor.decoder_bufr_files(str(self.filepath))
        for msg in messages:
            yield self.schema.map_observation(msg)


class _BUFRIterableDataset(IterableDataset):
    """Internal IterableDataset wrapper for PyTorch DataLoader."""

    def __init__(self, bufr_loader: BUFR_dataloader):
        self.bufr_loader = bufr_loader

    def __iter__(self):
        return iter(self.bufr_loader)
