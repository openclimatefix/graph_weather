import pytest
import numpy as np
from graph_weather.data.bufr_process import NNJA_Schema


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