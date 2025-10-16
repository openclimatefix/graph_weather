from dataclasses import dataclass, field
from typing import Optional, Callable, Any , List, Dict
import numpy as np
import logging 
import pandas as pd 

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


    def __init__(self):
        pass

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
            )
        }

class CRIS_Schema(DataSourceSchema):
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
        }
class BUFR_dataloader:
    def __init__(self,dataset,batch_size):
        """
            Args:
                -> dataset : str (path)
                -> batch_size : int
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
    def decoder(self):
        pass 
    def map_to_nnjai_schema(self):
        pass
    
    
class _BUFRIterableDataset(IterableDataset):
    """Internal IterableDataset wrapper for PyTorch DataLoader."""
    
    def __init__(self, bufr_loader: BUFR_dataLoader):
        self.bufr_loader = bufr_loader
    
    def __iter__(self):
        return iter(self.bufr_loader)

