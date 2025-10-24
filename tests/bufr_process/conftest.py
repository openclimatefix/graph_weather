import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph_weather.data.bufr_process import  DataSourceSchema

@pytest.fixture
def mock_schema():
    """Mock schema for testing."""
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
def sample_adpupa_data():
    """Sample ADPUPA test data."""
    return {
        'latitude': 45.0,
        'longitude': -120.0,
        'obsTime': '2023-01-01T12:00:00',
        'airTemperature': 300.0,
        'pressure': 101325.0,
        'height': 100.0,
        'dewpointTemperature': 290.0,
        'windU': 5.0,
        'windV': -3.0
    }

@pytest.fixture
def sample_cris_data():
    """Sample CrIS test data."""
    return {
        'latitude': 30.0,
        'longitude': -100.0,
        'obsTime': 1672574400,
        'retrievedTemperature': 280.0,
        'retrievedPressure': 85000.0,
        'qualityFlags': 1
    }