from dataclasses import dataclass, field
from typing import Optional, Callable, Any , List, Dict
import numpy as np
import logging 
import pandas as pd 
import xarray as xr
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Bufr Processor")

from torch.utils.data import DataLoader, IterableDataset

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
        'OBS_TIMESTAMP': 'datetime64[ns]',  
        'LAT': 'float32',                   
        'LON': 'float32',                    
    }

    VARIABLES = {
        'temperature': 'float32',
        'pressure': 'float32',
        'relative_humidity': 'float32',
        'u_wind': 'float32',
        'v_wind': 'float32',
        'dew_point': 'float32',
        'height': 'float32',
    }
    
    ATTRIBUTES = {
        'source': 'DATA_SOURCE',         
        'qc_flag': 'int8',                 
        'processing_timestamp': 'datetime64[ns]',
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
        required_coords = ['OBS_TIMESTAMP', 'LAT', 'LON']
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
        required = ['OBS_TIMESTAMP', 'LAT', 'LON']
        mapped_outputs = {m.output_name for m in self.field_mappings.values()}
        missing = [r for r in required if r not in mapped_outputs]
        if missing:
            logger.warning(
                f"{self.source_name} schema missing required outputs: {missing}"
            )
    
    def map_observation(self, bufr_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw BUFR message to NNJA-AI format.
        
        Args:
            bufr_message: Decoded BUFR message (dict of field → values)
        
        Returns:
            Observation in NNJA format (dict matching NNJASchema)
        """
        mapped = {}
        
        for field_map in self.field_mappings.values():
            if field_map.source_name in bufr_message:
                raw_value = bufr_message[field_map.source_name]
                try:
                    value = field_map.apply(raw_value)
                    mapped[field_map.output_name] = value
                except Exception as e:
                    logger.warning(
                        f"Error transforming {field_map.source_name}: {e}"
                    )
                    mapped[field_map.output_name] = None
        
        return mapped
    
    def get_variable_list(self) -> List[str]:
        """Get list of NNJA variables this source provides."""
        return list(set(m.output_name for m in self.field_mappings.values()))

class ADPUPA_schema(DataSourceSchema):
    """ADPUPA (upper-air radiosonde) BUFR schema mapping to NNJA-AI."""

    source_name = "ADPUPA"
    def _build_mappings(self):
        self.field_mappings= {
            'latitude' : FieldMapping(
                source_name='latitude',
                output_name='LAT',
                dtype=float,
                description='Station latitude'
            ),
            'longitude' : FieldMapping(
                source_name='longitude',
                output_name='LON',
                dtype=float,
                description='Station longitude'
            ),
            'obsTime' : FieldMapping(
                source_name='obsTime',
                output_name='OBS_TIME', 
                description='datetime64[ns]',
                transform_fn=lambda x: pd.Timestamp(x).value if isinstance(x, str) else x,
                description='Observation timestamp'
            ), 
            'airTemperature'  : FieldMapping(
                source_name='airTemperature',
                output_name='temperature',
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x, 
                description='Temperature in Celsius'
            ),
            'pressure' : FieldMapping(
                source_name='pressure',
                output_name='pressure',
                dtype=float,
                description='Pressure in Pa'
            ),
            'height' : FieldMapping(
                source_name='height',
                output_name='height',
                dtype=float,
                description='Height above sealevel in m'
            ),
            'dewpointTemperature' : FieldMapping(
                source_name='dewpointTemperature',
                output_name='dew_point',
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                description='Dew point in Celsius'
            ),
            'windU' : FieldMapping(
                source_name='windU',
                output_name='u_wind',
                dtype=float,
                description='U-component wind (m/s)'
            ),
            'windV': FieldMapping(
                source_name='windV',
                output_name='v_wind',
                dtype=float,
                description='V-component wind (m/s)'
            ),
            'stationId': FieldMapping(
                source_name='stationId',
                output_name='station_id',
                dtype=str,
                required=False,
                description='Station identifier'
            )
        }
    
    def _convert_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert BUFR timestamp to pandas Timestamp."""
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit='s')
        elif isinstance(value, str):
            return pd.Timestamp(value)
        else:
            return pd.Timestamp(value)


class CRIS_schema(DataSourceSchema):
    """CrIS (satellite hyperspectral) BUFR schema mapping to NNJA-AI."""
    
    source_name = "CrIS"
    def _build_mappings(self):
        self.field_mappings = {
            'latitude' : FieldMapping(
                source_name='lat',
                output_name='LAT',
                dtype=float
            ),
            'longitude' : FieldMapping(
                source_name='lon',
                output_name='LON',
                dtype=float
            ),
            'obsTime' : FieldMapping(
                source_name='obsTime',
                output_name='OBS_TIMESTAMP',
                dtype='datetime64[ns]',
                transform_fn=lambda x: pd.Timestamp(x).value,
            ),
            'retrievedTemperature': FieldMapping(
                source_name='retrievedTemperature',
                output_name='temperature',
                dtype=float,
                transform_fn=lambda x: x - 273.15,
            ),
            'retrievedPressure': FieldMapping(
                source_name='retrievedPressure',
                output_name='pressure',
                dtype=float,
            ),
            'sourceZenithAngle': FieldMapping(
                source_name='sensorZenithAngle',
                output_name='sensor_zenith_angle',
                dtype=float,
                required=False,
                description='Sensor zenith angle'
            ),
            'qualityFlags' : FieldMapping(
                source_name='qualityFlags',
                output_name='qc_flag',
                dtype=int,
                description='Quality control flags'
            )
        }
    def _convert_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert BUFR timestamp to pandas Timestamp."""
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit='s')
        elif isinstance(value, str):
            return pd.Timestamp(value)
        else:
            return pd.Timestamp(value)

class BUFR_processsor:
    """
    Low-level BUFR file decoder.
    Handles binary BUFR format decoding using eccodes library.
    """
    def __init__(self , schema : DataSourceSchema):
        """
            Args:
                -> schema : DataSourceSchema instance
        """
        if not isinstance(schema,DataSourceSchema):
            raise TypeError('schema must be of DataSourceSchema instance')
        
        self.schema = schema 
    
    def decoder_bufr_files(self, filepath) -> List[Dict[str,any]]:
        """
            Decode all messages from BUFR file.
            
            Args:
                -> filepath: Path to BUFR file
        """
        msgs = []
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"BUFR file not found: {filepath}")
        
        try: 
            with open(filepath, 'rb') as f: 
                while True:
                    bufr_id = eccodes.bufr_new_from_file(f)
                    if bufr_id is None:
                        break 
                    
                    try:
                        eccodes.codes_set(bufr_id, 'unpack', 1)
                        msg = {}
                        iterator = eccodes.bufr_keys_iterator(bufr_id)
                        while eccodes.bufr_keys_iterator_next(iterator):    
                            key = eccodes.bufr_keys_iterator_get_name(iterator)
                            try:
                                value = eccodes.codes_get_string(bufr_id, key)
                                msg[key] = value
                            except (eccodes.KeyValueNotFoundError, eccodes.CodesInternalError):
                                try:
                                    value = eccodes.codes_get_double(bufr_id, key)
                                    msg[key] = value
                                except (eccodes.KeyValueNotFoundError, eccodes.CodesInternalError):
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

    def process_files_to_dataframe(self, filepath : str)-> pd.DataFrame:
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
            mapped = self.schema.map_observations(msg)
            if mapped:
                transformed.append(mapped)
        
        if not transformed:
            logger.warning(f"No valid observations found in {filepath}")
            return pd.DataFrame()

        
        df = pd.DataFrame(transformed)
        
        for col in df.columns:
            if col in NNJA_Schema.COORDINATES:
                dtype = NNJA_Schema.COORDINATES[col]
                if 'datetime' in dtype:
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype.split('[')[0] if '[' in dtype else dtype)
        if not NNJA_Schema.validate_data(df.to_dict(orient='list')):
            logger.warning(f"DataFrame missing required NNJA coordinates from {filepath}")
        
        return df

    def process_files_to_xarray(self, filepath : str) -> xr.Dataset:
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
            if col not in ['OBS_TIMESTAMP', 'LAT', 'LON']:
                data_vars[col] = (['observation'], df[col].values)
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'obs' : df.index ,
                'time' :  ('obs', df['obs'].values),
                'lat' : ('obs', df['LAT'].values),
                'lon' : ('obs', df['LON'].values),
            }
        )
        ds.attrs['source'] = self.schema.source_name
        ds.attrs['processing_timestamp'] = pd.Timestamp.now().isoformat()
        ds.attrs['num_observations'] = len(df)
        
        return ds
      
    def process_files_to_parquet(self, filepath: str, output_path: str)->None:
        """
        Process BUFR file and save as Parquet in NNJA-AI format.
        
        Args:
            -> filepath: Path to BUFR file
            -> output_path: Path for output Parquet file
        """
        df = self.process_file_to_dataframe(filepath=filepath)
        
        if not df.empty:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(df)} observations to {output_path}")
        else:
            logger.warning(f"No data to save for {filepath}")

      
class BUFR_dataloader:  
    
    SCHEMA_REGISTRY={
        'ADPUPA': ADPUPA_schema,
        'CrIS': CRIS_schema,
    }
    def __init__(self,dataset:str,batch_size:int=32,schema_name: Optional[str] = None):
        """
            Args:
                -> dataset : str (path)
                -> batch_size : int
                -> schema_name : Data source name ('ADPUPA', 'CrIS', etc.)
                    If None, attempts to infer from filename
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        if schema_name is None:
            schema_name = self._infer_schema_from_path(dataset)
        
        if schema_name not in self.SCHEMA_REGISTRY:
            raise ValueError(
                f'Unknown schema "{schema_name}"\nAvailable : {list(self.SCHEMA_REGISTRY)}'
            )
        
    def decoder(self):
        pass 
    def map_to_nnjai_schema(self):
        pass
    def to_dataframe(self):
        pass 
    def to_parquet(self):
        pass 
    def __iter__(self):
        pass
    def get_dataloader(self):
        pass
    
class _BUFRIterableDataset(IterableDataset):
    """Internal IterableDataset wrapper for PyTorch DataLoader."""
    
    def __init__(self, bufr_loader: BUFR_dataloader):
        self.bufr_loader = bufr_loader
    
    def __iter__(self):
        return iter(self.bufr_loader)

