"""
Unit tests for the `SensorDataset` class, mocking the `DataCatalog` to simulate sensor data loading and validate dataset behavior. 

The tests ensure correct handling of data types, shapes, and batch processing for various sensor types.
"""

from datetime import datetime 
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
import pandas as pd

from graph_weather.data.nnja_ai import SensorDataset, collate_fn

def get_sensor_variables(sensor_type):
    """Helper function to get the correct variables for each sensor type."""
    if sensor_type == "AMSU":
        return [f"TMBR_000{i:02d}" for i in range(1, 16)]  # 15 channels
    elif sensor_type == "ATMS":
        return [f"TMBR_000{i:02d}" for i in range(1, 23)]  # 22 channels
    elif sensor_type == "MHS":
        return [f"TMBR_000{i:02d}" for i in range(1, 6)]   # 5 channels
    elif sensor_type == "IASI":
        return [f"SCRA_{str(i).zfill(5)}" for i in range(1, 617)]  # 616 channels
    elif sensor_type == "CrIS":
        return [f"SRAD01_{str(i).zfill(5)}" for i in range(1, 432)]  # 431 channels
    return []

@pytest.fixture
def mock_datacatalog():
    """
    Fixture to mock the DataCatalog for unit tests to avoid actual data loading.
    """
    with patch("graph_weather.data.nnja_ai.DataCatalog") as mock:
        # Create a mock catalog
        mock_catalog = MagicMock()
        
        # Create a mock dataset with direct DataFrame return
        mock_dataset = MagicMock()
        mock_dataset.load_manifest = MagicMock()
        mock_dataset.sel = MagicMock(return_value=mock_dataset)  # Return self to chain calls
        
        def create_mock_df(engine="pandas"):
            # Get the sensor type from the mock dataset
            sensor_vars = get_sensor_variables(mock_dataset.sensor_type)
            
            # Create DataFrame with required columns
            df = pd.DataFrame({
                "OBS_TIMESTAMP": pd.date_range(start=datetime(2021, 1, 1), periods=100, freq='H'),
                "LAT": np.full(100, 45.0),
                "LON": np.full(100, -120.0)
            })
            
            # Add sensor-specific variables
            for var in sensor_vars:
                df[var] = np.full(100, 250.0)
            
            return df
        
        # Set up the mock to return our DataFrame
        mock_dataset.load_dataset = create_mock_df
        
        # Configure the catalog to return our mock dataset
        def get_mock_dataset(self, name):
            # Set the sensor type based on the requested dataset name
            mock_dataset.sensor_type = next(
                config["sensor_type"] for config in SENSOR_CONFIGS 
                if config["name"] == name
            )
            return mock_dataset
        
        mock_catalog.__getitem__ = get_mock_dataset  # Fix: Explicitly define the method with `self`
        mock.return_value = mock_catalog
        
        yield mock

# Test configurations
SENSOR_CONFIGS = [
    {
        "name": "amsu-1bamua-NC021023",
        "sensor_type": "AMSU",
        "expected_metadata_size": 15  # 15 TMBR channels
    },
    {
        "name": "atms-atms-NC021203",
        "sensor_type": "ATMS",
        "expected_metadata_size": 22  # 22 TMBR channels
    },
    {
        "name": "mhs-1bmhs-NC021027",
        "sensor_type": "MHS",
        "expected_metadata_size": 5   # 5 TMBR channels
    },
    {
        "name": "iasi-mtiasi-NC021241",
        "sensor_type": "IASI",
        "expected_metadata_size": 616  # 616 SCRA channels
    },
    {
        "name": "cris-crisf4-NC021206",
        "sensor_type": "CrIS",
        "expected_metadata_size": 431  # 431 SRAD channels
    }
]

@pytest.mark.parametrize("sensor_config", SENSOR_CONFIGS)
def test_sensor_dataset(mock_datacatalog, sensor_config):
    """Test the SensorDataset class for different sensor types."""
    time = datetime(2021, 1, 1, 0, 0)
    primary_descriptors = ["OBS_TIMESTAMP", "LAT", "LON"]
    
    dataset = SensorDataset(
        dataset_name=sensor_config["name"],
        time=time,
        primary_descriptors=primary_descriptors,
        additional_variables=get_sensor_variables(sensor_config["sensor_type"]),
        sensor_type=sensor_config["sensor_type"]
    )

    # Test dataset length
    assert len(dataset) > 0, f"Dataset should not be empty for {sensor_config['sensor_type']}"

    # Test single item structure
    item = dataset[0]
    expected_keys = {"timestamp", "latitude", "longitude", "metadata"}
    assert set(item.keys()) == expected_keys, f"Dataset item keys are not as expected for {sensor_config['sensor_type']}"

    # Validate tensor properties
    assert isinstance(item["timestamp"], torch.Tensor), f"Timestamp should be a tensor for {sensor_config['sensor_type']}"
    assert item["timestamp"].dtype == torch.float32, f"Timestamp should have dtype float32 for {sensor_config['sensor_type']}"
    assert item["timestamp"].ndim == 0, f"Timestamp should be a scalar tensor for {sensor_config['sensor_type']}"

    assert isinstance(item["latitude"], torch.Tensor), f"Latitude should be a tensor for {sensor_config['sensor_type']}"
    assert item["latitude"].dtype == torch.float32, f"Latitude should have dtype float32 for {sensor_config['sensor_type']}"
    assert item["latitude"].ndim == 0, f"Latitude should be a scalar tensor for {sensor_config['sensor_type']}"

    assert isinstance(item["longitude"], torch.Tensor), f"Longitude should be a tensor for {sensor_config['sensor_type']}"
    assert item["longitude"].dtype == torch.float32, f"Longitude should have dtype float32 for {sensor_config['sensor_type']}"
    assert item["longitude"].ndim == 0, f"Longitude should be a scalar tensor for {sensor_config['sensor_type']}"

    assert isinstance(item["metadata"], torch.Tensor), f"Metadata should be a tensor for {sensor_config['sensor_type']}"
    assert item["metadata"].shape == (sensor_config["expected_metadata_size"],), \
        f"Metadata shape mismatch for {sensor_config['sensor_type']}. Expected ({sensor_config['expected_metadata_size']},)"
    assert item["metadata"].dtype == torch.float32, f"Metadata should have dtype float32 for {sensor_config['sensor_type']}"


def test_collate_function():
    """Test the collate_fn function to ensure proper batching of dataset items."""
    batch_size = 4
    metadata_size = 15  # Using AMSU size for this test
    mock_batch = [
        {
            "timestamp": torch.tensor(datetime.now().timestamp(), dtype=torch.float32),
            "latitude": torch.tensor(45.0, dtype=torch.float32),
            "longitude": torch.tensor(-120.0, dtype=torch.float32),
            "metadata": torch.randn(metadata_size, dtype=torch.float32),
        }
        for _ in range(batch_size)
    ]

    batched = collate_fn(mock_batch)

    assert batched["timestamp"].shape == (batch_size,), "Timestamp batch shape mismatch"
    assert batched["latitude"].shape == (batch_size,), "Latitude batch shape mismatch"
    assert batched["longitude"].shape == (batch_size,), "Longitude batch shape mismatch"
    assert batched["metadata"].shape == (batch_size, metadata_size), "Metadata batch shape mismatch"
    assert batched["timestamp"].dtype == torch.float32, "Timestamp dtype mismatch"
    assert batched["latitude"].dtype == torch.float32, "Latitude dtype mismatch"
    assert batched["longitude"].dtype == torch.float32, "Longitude dtype mismatch"
    assert batched["metadata"].dtype == torch.float32, "Metadata dtype mismatch"
