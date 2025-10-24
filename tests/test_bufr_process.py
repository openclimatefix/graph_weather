import pytest
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

try:
    from graph_weather.data.bufr_process import (
        FieldMapping,
        NNJA_Schema,
        DataSourceSchema,
        ADPUPA_schema,
        CRIS_schema,
        BUFR_processor,
        BUFR_dataloader
    )
except ImportError:
    # Fallback to direct import
    from data.bufr_process import (
        FieldMapping,
        NNJA_Schema,
        DataSourceSchema,
        ADPUPA_schema,
        CRIS_schema,
        BUFR_processor,
        BUFR_dataloader
    )


class TestFieldMapping:
    """Test FieldMapping dataclass."""
    
    def test_field_mapping_creation(self):
        """Test FieldMapping initialization."""
        mapping = FieldMapping(
            source_name="temperature",
            output_name="temp",
            dtype=float,
            description="Temperature field"
        )
        
        assert mapping.source_name == "temperature"
        assert mapping.output_name == "temp"
        assert mapping.dtype == float
        assert mapping.description == "Temperature field"
        assert mapping.required is True
        assert mapping.transform_fn is None
    
    def test_field_mapping_apply_no_transform(self):
        """Test apply method without transformation function."""
        mapping = FieldMapping(
            source_name="pressure",
            output_name="pres",
            dtype=float
        )
        
        result = mapping.apply(1013.25)
        assert result == 1013.25
    
    def test_field_mapping_apply_with_transform(self):
        """Test apply method with transformation function."""
        mapping = FieldMapping(
            source_name="temp_k",
            output_name="temp_c",
            dtype=float,
            transform_fn=lambda x: x - 273.15
        )
        
        result = mapping.apply(300.0)
        assert result == pytest.approx(26.85)
    
    def test_field_mapping_apply_none_value(self):
        """Test apply method with None value."""
        mapping = FieldMapping(
            source_name="missing",
            output_name="missing_out",
            dtype=float
        )
        
        result = mapping.apply(None)
        assert result is None


class TestNNJASchema:
    """Test NNJA_Schema class."""
    
    def test_schema_structure(self):
        """Test schema has expected structure."""
        assert 'OBS_TIMESTAMP' in NNJA_Schema.COORDINATES
        assert 'LAT' in NNJA_Schema.COORDINATES
        assert 'LON' in NNJA_Schema.COORDINATES
        assert 'temperature' in NNJA_Schema.VARIABLES
        assert 'pressure' in NNJA_Schema.VARIABLES
    
    def test_to_xarray_schema(self):
        """Test schema combination."""
        full_schema = NNJA_Schema.to_xarray_schema()
        assert 'OBS_TIMESTAMP' in full_schema
        assert 'temperature' in full_schema
        assert 'source' in full_schema
    
    def test_get_coordinate_names(self):
        """Test coordinate names retrieval."""
        coords = NNJA_Schema.get_coordinate_names()
        expected_coords = ['OBS_TIMESTAMP', 'LAT', 'LON']
        assert set(coords) == set(expected_coords)
    
    def test_validate_data(self):
        """Test data validation."""
        valid_data = {
            'OBS_TIMESTAMP': np.array(['2023-01-01'], dtype='datetime64[ns]'),
            'LAT': np.array([45.0], dtype='float32'),
            'LON': np.array([-120.0], dtype='float32')
        }
        assert NNJA_Schema.validate_data(valid_data) is True
        
        invalid_data = {
            'LAT': np.array([45.0], dtype='float32'),
            'LON': np.array([-120.0], dtype='float32')
        }
        assert NNJA_Schema.validate_data(invalid_data) is False


class MockADPUPASchema(ADPUPA_schema):
    """Mock ADPUPA schema with proper FieldMapping dtypes."""
    
    def _build_mappings(self):
        self.field_mappings = {
            'latitude': FieldMapping(
                source_name='latitude',
                output_name='LAT',
                dtype=float,
                description='Station latitude'
            ),
            'longitude': FieldMapping(
                source_name='longitude',
                output_name='LON',
                dtype=float,
                description='Station longitude'
            ),
            'obsTime': FieldMapping(
                source_name='obsTime',
                output_name='OBS_TIMESTAMP',
                dtype=object,
                transform_fn=self._convert_timestamp,
                description='Observation timestamp'
            ), 
            'airTemperature': FieldMapping(
                source_name='airTemperature',
                output_name='temperature',
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x, 
                description='Temperature in Celsius'
            ),
            'pressure': FieldMapping(
                source_name='pressure',
                output_name='pressure',
                dtype=float,
                description='Pressure in Pa'
            ),
            'height': FieldMapping(
                source_name='height',
                output_name='height',
                dtype=float,
                description='Height above sealevel in m'
            ),
            'dewpointTemperature': FieldMapping(
                source_name='dewpointTemperature',
                output_name='dew_point',
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                description='Dew point in Celsius'
            ),
            'windU': FieldMapping(
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


class MockCRISSchema(CRIS_schema):
    """Mock CrIS schema with proper FieldMapping dtypes."""
    
    def _build_mappings(self):
        self.field_mappings = {
            'latitude': FieldMapping(
                source_name='latitude',
                output_name='LAT',
                dtype=float,
                description='Satellite latitude'
            ),
            'longitude': FieldMapping(
                source_name='longitude',
                output_name='LON',
                dtype=float,
                description='Satellite longitude'
            ),
            'obsTime': FieldMapping(
                source_name='obsTime',
                output_name='OBS_TIMESTAMP',
                dtype=object,
                transform_fn=self._convert_timestamp,
                description='Observation timestamp'
            ),
            'retrievedTemperature': FieldMapping(
                source_name='retrievedTemperature',
                output_name='temperature',
                dtype=float,
                transform_fn=lambda x: x - 273.15 if x > 100 else x,
                description='Retrieved temperature in Celsius'
            ),
            'retrievedPressure': FieldMapping(
                source_name='retrievedPressure',
                output_name='pressure',
                dtype=float,
                description='Retrieved pressure in Pa'
            ),
            'sensorZenithAngle': FieldMapping(
                source_name='sensorZenithAngle',
                output_name='sensor_zenith_angle',
                dtype=float,
                required=False,
                description='Sensor zenith angle'
            ),
            'qualityFlags': FieldMapping(
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


class TestADPUPASchema:
    """Test ADPUPA schema implementation."""
    
    @pytest.fixture
    def adpupa_schema(self):
        return MockADPUPASchema()
    
    def test_schema_creation(self, adpupa_schema):
        """Test ADPUPA schema initialization."""
        assert adpupa_schema.source_name == "ADPUPA"
        assert len(adpupa_schema.field_mappings) > 0
    
    def test_required_mappings_present(self, adpupa_schema):
        """Test required NNJA coordinates are mapped."""
        output_names = {m.output_name for m in adpupa_schema.field_mappings.values()}
        assert 'LAT' in output_names
        assert 'LON' in output_names
        assert 'OBS_TIMESTAMP' in output_names
    
    def test_map_observation(self, adpupa_schema):
        """Test observation mapping."""
        test_message = {
            'latitude': 45.0,
            'longitude': -120.0,
            'obsTime': '2023-01-01T12:00:00',
            'airTemperature': 300.0,  # Kelvin
            'pressure': 101325.0,
            'height': 100.0,
            'dewpointTemperature': 290.0,  # Kelvin
            'windU': 5.0,
            'windV': -3.0
        }
        
        mapped = adpupa_schema.map_observation(test_message)
        
        assert mapped['LAT'] == 45.0
        assert mapped['LON'] == -120.0
        assert isinstance(mapped['OBS_TIMESTAMP'], pd.Timestamp)
        assert mapped['temperature'] == pytest.approx(26.85)  # 300K to C
        assert mapped['pressure'] == 101325.0
        assert mapped['dew_point'] == pytest.approx(16.85)  # 290K to C
    
    def test_map_observation_missing_fields(self, adpupa_schema):
        """Test mapping with missing fields."""
        test_message = {
            'latitude': 45.0,
            'longitude': -120.0,
            'obsTime': '2023-01-01T12:00:00'
        }
        
        mapped = adpupa_schema.map_observation(test_message)
        
        assert mapped['LAT'] == 45.0
        assert mapped['LON'] == -120.0
        assert mapped['temperature'] is None  # Missing field


class TestCRISSchema:
    """Test CrIS schema implementation."""
    
    @pytest.fixture
    def cris_schema(self):
        return MockCRISSchema()
    
    def test_schema_creation(self, cris_schema):
        """Test CrIS schema initialization."""
        assert cris_schema.source_name == "CrIS"
        assert len(cris_schema.field_mappings) > 0
    
    def test_map_observation(self, cris_schema):
        """Test CrIS observation mapping."""
        test_message = {
            'latitude': 30.0,
            'longitude': -100.0,
            'obsTime': 1672574400,  # Unix timestamp
            'retrievedTemperature': 280.0,  # Kelvin
            'retrievedPressure': 85000.0,
            'qualityFlags': 1
        }
        
        mapped = cris_schema.map_observation(test_message)
        
        assert mapped['LAT'] == 30.0
        assert mapped['LON'] == -100.0
        assert isinstance(mapped['OBS_TIMESTAMP'], pd.Timestamp)
        assert mapped['temperature'] == pytest.approx(6.85)  # 280K to C
        assert mapped['pressure'] == 85000.0
        assert mapped['qc_flag'] == 1


class TestBUFRProcessor:
    """Test BUFR_processor class."""
    
    @pytest.fixture
    def mock_schema(self):
        schema = Mock(spec=DataSourceSchema)
        schema.source_name = "TEST"
        schema.field_mappings = {}
        schema.map_observation.return_value = {
            'OBS_TIMESTAMP': pd.Timestamp('2023-01-01T12:00:00'),
            'LAT': 45.0,
            'LON': -120.0,
            'temperature': 20.0
        }
        return schema
    
    @pytest.fixture
    def bufr_processor(self, mock_schema):
        return BUFR_processor(mock_schema)
    
    def test_processor_initialization(self, mock_schema):
        """Test processor initialization."""
        processor = BUFR_processor(mock_schema)
        assert processor.schema == mock_schema
    
    def test_processor_invalid_schema(self):
        """Test processor with invalid schema."""
        with pytest.raises(TypeError):
            BUFR_processor("invalid_schema")
    
    @patch('graph_weather.data.bufr_process.eccodes', create=True)
    def test_decode_bufr_file_success(self, mock_eccodes, bufr_processor, tmp_path):
        """Test successful BUFR file decoding."""
        # Create a temporary BUFR file
        test_file = tmp_path / "test.bufr"
        test_file.write_bytes(b"test bufr content")
        
        # Mock eccodes behavior
        mock_bufr_id = Mock()
        mock_eccodes.codes_bufr_new_from_file.side_effect = [mock_bufr_id, None]
        mock_iterator = Mock()
        mock_eccodes.codes_bufr_keys_iterator_new.return_value = mock_iterator
        mock_eccodes.codes_bufr_keys_iterator_next.side_effect = [True, False]
        mock_eccodes.codes_bufr_keys_iterator_get_name.return_value = "test_key"
        mock_eccodes.codes_get_string.return_value = "test_value"
        
        # Use the correct method name - decoder_bufr_files
        messages = bufr_processor.decoder_bufr_files(str(test_file))
        
        assert len(messages) == 1
        assert messages[0]["test_key"] == "test_value"
    
    def test_decode_bufr_file_not_found(self, bufr_processor, tmp_path):
        """Test BUFR file not found."""
        with pytest.raises(FileNotFoundError):
            bufr_processor.decoder_bufr_files(str(tmp_path / "nonexistent.bufr"))
    
    @patch.object(BUFR_processor, 'decoder_bufr_files')
    def test_process_files_to_dataframe(self, mock_decode, bufr_processor, mock_schema):
        """Test processing BUFR file to DataFrame."""
        # Mock decoded messages
        mock_messages = [
            {'latitude': 45.0, 'longitude': -120.0, 'obsTime': '2023-01-01T12:00:00'},
            {'latitude': 46.0, 'longitude': -121.0, 'obsTime': '2023-01-01T12:30:00'}
        ]
        mock_decode.return_value = mock_messages
        
        # Mock schema mapping
        mock_schema.map_observation.side_effect = [
            {'OBS_TIMESTAMP': pd.Timestamp('2023-01-01T12:00:00'), 'LAT': 45.0, 'LON': -120.0},
            {'OBS_TIMESTAMP': pd.Timestamp('2023-01-01T12:30:00'), 'LAT': 46.0, 'LON': -121.0}
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            df = bufr_processor.process_files_to_dataframe(f.name)
        
        assert len(df) == 2
        assert 'OBS_TIMESTAMP' in df.columns
        assert 'LAT' in df.columns
        assert 'LON' in df.columns
        assert df['LAT'].iloc[0] == 45.0
    
    @patch.object(BUFR_processor, 'decoder_bufr_files')
    def test_process_files_to_dataframe_empty(self, mock_decode, bufr_processor):
        """Test processing BUFR file with no valid observations."""
        mock_decode.return_value = []  # No messages
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            df = bufr_processor.process_files_to_dataframe(f.name)
        
        assert df.empty
    
    @patch.object(BUFR_processor, 'process_files_to_dataframe')
    def test_process_files_to_xarray(self, mock_process, bufr_processor):
        """Test processing BUFR file to xarray Dataset."""
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'OBS_TIMESTAMP': [pd.Timestamp('2023-01-01T12:00:00'), pd.Timestamp('2023-01-01T12:30:00')],
            'LAT': [45.0, 46.0],
            'LON': [-120.0, -121.0],
            'temperature': [20.0, 19.5],
            'pressure': [101325.0, 101300.0]
        })
        mock_process.return_value = mock_df
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            ds = bufr_processor.process_files_to_xarray(f.name)
        
        assert 'temperature' in ds.data_vars
        assert 'pressure' in ds.data_vars
        assert 'time' in ds.coords
        assert 'lat' in ds.coords
        assert 'lon' in ds.coords
        assert ds.attrs['source'] == 'TEST'
    
    @patch.object(BUFR_processor, 'process_files_to_dataframe')
    def test_process_files_to_parquet(self, mock_process, bufr_processor, tmp_path):
        """Test processing BUFR file to Parquet."""
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'OBS_TIMESTAMP': [pd.Timestamp('2023-01-01T12:00:00')],
            'LAT': [45.0],
            'LON': [-120.0],
            'temperature': [20.0]
        })
        mock_process.return_value = mock_df
        
        output_file = tmp_path / "output.parquet"
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            bufr_processor.process_files_to_parquet(f.name, str(output_file))
        
        assert output_file.exists()


class TestBUFRDataLoader:
    """Test BUFR_dataloader class."""
    
    def test_dataloader_initialization_with_schema(self):
        """Test dataloader initialization with explicit schema."""
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            loader = BUFR_dataloader(f.name, schema_name='ADPUPA')
        
        assert loader.schema_name == 'ADPUPA'
        assert isinstance(loader.schema, ADPUPA_schema)
        assert isinstance(loader.processor, BUFR_processor)
    
    def test_dataloader_initialization_infer_schema(self):
        """Test dataloader initialization with schema inference."""
        test_cases = [
            ('test_adpupa.bufr', 'ADPUPA'),
            ('test_CRIS_data.bufr', 'CrIS'),
            ('unknown_file.bufr', 'ADPUPA')  # Default case
        ]
        
        for filename, expected_schema in test_cases:
            with tempfile.NamedTemporaryFile(suffix=filename) as f:
                loader = BUFR_dataloader(f.name)
                assert loader.schema_name == expected_schema
    
    def test_dataloader_initialization_invalid_schema(self):
        """Test dataloader initialization with invalid schema."""
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            with pytest.raises(ValueError, match='Unknown schema "INVALID"'):
                BUFR_dataloader(f.name, schema_name='INVALID')
    
    @patch.object(BUFR_processor, 'process_files_to_dataframe')
    def test_to_dataframe(self, mock_process):
        """Test to_dataframe method."""
        mock_df = pd.DataFrame({
            'OBS_TIMESTAMP': [pd.Timestamp('2023-01-01T12:00:00')],
            'LAT': [45.0],
            'LON': [-120.0]
        })
        mock_process.return_value = mock_df
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            loader = BUFR_dataloader(f.name, schema_name='ADPUPA')
            df = loader.to_dataframe()
        
        assert len(df) == 1
        mock_process.assert_called_once_with(str(Path(f.name)))
    
    @patch.object(BUFR_processor, 'process_files_to_xarray')
    def test_to_xarray(self, mock_process):
        """Test to_xarray method."""
        mock_ds = xr.Dataset({
            'temperature': (['obs'], [20.0]),
            'pressure': (['obs'], [101325.0])
        }, coords={
            'obs': [0],
            'time': ('obs', [pd.Timestamp('2023-01-01T12:00:00')]),
            'lat': ('obs', [45.0]),
            'lon': ('obs', [-120.0])
        })
        mock_process.return_value = mock_ds
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            loader = BUFR_dataloader(f.name, schema_name='ADPUPA')
            ds = loader.to_xarray()
        
        assert 'temperature' in ds.data_vars
        mock_process.assert_called_once_with(str(Path(f.name)))
    
    @patch.object(BUFR_processor, 'process_files_to_parquet')
    def test_to_parquet(self, mock_process, tmp_path):
        """Test to_parquet method."""
        output_file = tmp_path / "test_output.parquet"
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            loader = BUFR_dataloader(f.name, schema_name='ADPUPA')
            loader.to_parquet(str(output_file))
        
        mock_process.assert_called_once_with(str(Path(f.name)), str(output_file))
    
    @patch.object(BUFR_processor, 'decoder_bufr_files')
    def test_iterator(self, mock_decode):
        """Test dataloader iterator."""
        mock_messages = [
            {'lat': 45.0, 'lon': -120.0},
            {'lat': 46.0, 'lon': -121.0}
        ]
        mock_decode.return_value = mock_messages
        
        with tempfile.NamedTemporaryFile(suffix='.bufr') as f:
            loader = BUFR_dataloader(f.name, schema_name='ADPUPA')
            
            # Mock the schema's map_observation to return simple data
            loader.schema.map_observation = lambda x: {'LAT': x['lat'], 'LON': x['lon']}
            
            observations = list(loader)
        
        assert len(observations) == 2
        assert observations[0]['LAT'] == 45.0
        assert observations[1]['LAT'] == 46.0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_schema_registry_completeness(self):
        """Test that all schemas in registry can be instantiated."""
        for schema_name, schema_class in BUFR_dataloader.SCHEMA_REGISTRY.items():
            schema = schema_class()
            assert isinstance(schema, DataSourceSchema)
            assert schema.source_name == schema_name
    
    def test_end_to_end_mock_processing(self):
        """Test complete mock processing pipeline."""
        with tempfile.NamedTemporaryFile(suffix='_adpupa.bufr') as f:
            loader = BUFR_dataloader(f.name)  # Should infer ADPUPA schema
            
            assert loader.schema_name == 'ADPUPA'
            assert isinstance(loader.schema, ADPUPA_schema)
            assert isinstance(loader.processor, BUFR_processor)
            assert loader.processor.schema == loader.schema


# Test configuration for running with different options
def pytest_configure(config):
    """Pytest configuration hook."""
    print("Setting up BUFR processor tests...")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])