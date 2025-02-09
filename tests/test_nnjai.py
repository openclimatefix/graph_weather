"""
Tests for the nnjai_wrapp module in the graph_weather package.

This file contains unit tests for AMSUDataset and collate_fn functions.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from graph_weather.data.nnja_ai import NNJADataset, collate_fn

@pytest.fixture
def mock_datacatalog():
    """
    Fixture to mock the DataCatalog for unit tests to avoid actual data loading.
    This mock provides a mock dataset with predefined columns and values.
    """
    with patch("graph_weather.data.nnjai_wrapp.DataCatalog") as mock:
        # Mock dataset structure
        mock_df = MagicMock()
        mock_df.columns = ["OBS_TIMESTAMP", "LAT", "LON", "TMBR_00001", "TMBR_00002"]

        # Define a mock row
        class MockRow:
            def __getitem__(self, key):
                data = {
                    "OBS_TIMESTAMP": datetime.now(),
                    "LAT": 45.0,
                    "LON": -120.0,
                    "TMBR_00001": 250.0,
                    "TMBR_00002": 260.0,
                }
                return data.get(key, None)

        # Configure mock dataset
        mock_row = MockRow()
        mock_df.iloc = MagicMock()
        mock_df.iloc.__getitem__.return_value = mock_row
        mock_df.__len__.return_value = 100

        mock_dataset = MagicMock()
        mock_dataset.load_dataset.return_value = mock_df
        mock_dataset.sel.return_value = mock_dataset
        mock_dataset.load_manifest = MagicMock()

        mock.return_value.__getitem__.return_value = mock_dataset
        yield mock


def test_sensor_dataset(mock_datacatalog):
    """
    Test the SensorDataset class to ensure proper data loading and tensor structure for different sensors.
    """
    # Test for AMSU dataset
    dataset_name = "amsu-1bamua-NC021023"
    time = datetime(2021, 1, 1, 0, 0)  # Using datetime object instead of string
    primary_descriptors = ["OBS_TIMESTAMP", "LAT", "LON"]
    additional_variables = ["TMBR_00001", "TMBR_00002"]
    dataset = SensorDataset(dataset_name, time, primary_descriptors, additional_variables, sensor_type="AMSU")

    # Test dataset length
    assert len(dataset) > 0, "Dataset should not be empty."

    item = dataset[0]
    expected_keys = {"timestamp", "latitude", "longitude", "metadata"}
    assert set(item.keys()) == expected_keys, "Dataset item keys are not as expected."

    # Validate tensor properties
    assert isinstance(item["timestamp"], torch.Tensor), "Timestamp should be a tensor."
    assert item["timestamp"].dtype == torch.float32, "Timestamp should have dtype float32."
    assert item["timestamp"].ndim == 0, "Timestamp should be a scalar tensor."
    assert isinstance(item["latitude"], torch.Tensor), "Latitude should be a tensor."
    assert item["latitude"].dtype == torch.float32, "Latitude should have dtype float32."
    assert item["latitude"].ndim == 0, "Latitude should be a scalar tensor."
    assert isinstance(item["longitude"], torch.Tensor), "Longitude should be a tensor."
    assert item["longitude"].dtype == torch.float32, "Longitude should have dtype float32."
    assert item["longitude"].ndim == 0, "Longitude should be a scalar tensor."
    assert isinstance(item["metadata"], torch.Tensor), "Metadata should be a tensor."
    assert item["metadata"].shape == (len(additional_variables),), f"Metadata shape mismatch. Expected ({len(additional_variables)},)."
    assert item["metadata"].dtype == torch.float32, "Metadata should have dtype float32."


def test_collate_function():
    """
    Test the collate_fn function to ensure proper batching of dataset items.
    """
    # Mock a batch of items
    batch_size = 4
    metadata_size = 2
    mock_batch = [
        {
            "timestamp": torch.tensor(datetime.now().timestamp(), dtype=torch.float32),
            "latitude": torch.tensor(45.0, dtype=torch.float32),
            "longitude": torch.tensor(-120.0, dtype=torch.float32),
            "metadata": torch.randn(metadata_size, dtype=torch.float32),
        }
        for _ in range(batch_size)
    ]

    # Collate the batch
    batched = collate_fn(mock_batch)

    # Validate batched shapes and types
    assert batched["timestamp"].shape == (batch_size,), "Timestamp batch shape mismatch."
    assert batched["latitude"].shape == (batch_size,), "Latitude batch shape mismatch."
    assert batched["longitude"].shape == (batch_size,), "Longitude batch shape mismatch."
    assert batched["metadata"].shape == (batch_size, metadata_size), "Metadata batch shape mismatch."
    assert batched["timestamp"].dtype == torch.float32, "Timestamp dtype mismatch."
    assert batched["latitude"].dtype == torch.float32, "Latitude dtype mismatch."
    assert batched["longitude"].dtype == torch.float32, "Longitude dtype mismatch."
    assert batched["metadata"].dtype == torch.float32, "Metadata dtype mismatch."


def test_sensor_datasets(mock_datacatalog):
    """
    Test various sensor datasets (AMSU-A, ATMS, MHS, IASI, CrIS) to ensure they load properly
    and print the relevant information.
    """
    # Define datasets and associated parameters for different sensors
    sensors = [
        {"name": "amsu-1bamua-NC021023", "time": datetime(2021, 1, 1, 0, 0), "primary_descriptors": ["OBS_TIMESTAMP", "LAT", "LON"], "additional_variables": ["TMBR_00001", "TMBR_00002"], "sensor_type": "AMSU"},
        {"name": "atms-atms-NC021203", "time": datetime(2021, 1, 1, 0, 0), "primary_descriptors": ["OBS_TIMESTAMP", "LAT", "LON"], "additional_variables": ["TMBR_00001", "TMBR_00002"], "sensor_type": "ATMS"},
        {"name": "mhs-1bmhs-NC021027", "time": datetime(2021, 1, 1, 0, 0), "primary_descriptors": ["OBS_TIMESTAMP", "LAT", "LON"], "additional_variables": ["TMBR_00001", "TMBR_00002"], "sensor_type": "MHS"},
        {"name": "iasi-mtiasi-NC021241", "time": datetime(2021, 1, 1, 0, 0), "primary_descriptors": ["OBS_TIMESTAMP", "LAT", "LON"], "additional_variables": ["IASIL1CB"], "sensor_type": "IASI"},
        {"name": "cris-crisf4-NC021206", "time": datetime(2021, 1, 1, 0, 0), "primary_descriptors": ["OBS_TIMESTAMP", "LAT", "LON"], "additional_variables": ["SRAD01_00001", "SRAD01_00002"], "sensor_type": "CrIS"}
    ]

    # Loop through each sensor and load the dataset
    for sensor in sensors:
        print(f"\nTesting sensor: {sensor['name']}")

        # Create the dataset instance
        dataset = SensorDataset(sensor['name'], sensor['time'], sensor['primary_descriptors'], sensor['additional_variables'], sensor_type=sensor['sensor_type'])

        # Print dataset length
        print(f"Dataset length for {sensor['name']}: {len(dataset)}")

        # Retrieve and print the first item
        item = dataset[0]
        print(f"First item from {sensor['name']}:")
        print(item)

        # Ensure the dataset item structure is correct
        expected_keys = {"timestamp", "latitude", "longitude", "metadata"}
        assert set(item.keys()) == expected_keys, f"Dataset item keys for {sensor['name']} are not as expected."

        # Validate tensor properties
        assert isinstance(item["timestamp"], torch.Tensor), f"Timestamp should be a tensor for {sensor['name']}."
        assert item["timestamp"].dtype == torch.float32, f"Timestamp should have dtype float32 for {sensor['name']}."
        assert item["timestamp"].ndim == 0, f"Timestamp should be a scalar tensor for {sensor['name']}."

        assert isinstance(item["latitude"], torch.Tensor), f"Latitude should be a tensor for {sensor['name']}."
        assert item["latitude"].dtype == torch.float32, f"Latitude should have dtype float32 for {sensor['name']}."
        assert item["latitude"].ndim == 0, f"Latitude should be a scalar tensor for {sensor['name']}."

        assert isinstance(item["longitude"], torch.Tensor), f"Longitude should be a tensor for {sensor['name']}."
        assert item["longitude"].dtype == torch.float32, f"Longitude should have dtype float32 for {sensor['name']}."
        assert item["longitude"].ndim == 0, f"Longitude should be a scalar tensor for {sensor['name']}."

        assert isinstance(item["metadata"], torch.Tensor), f"Metadata should be a tensor for {sensor['name']}."
        assert item["metadata"].dtype == torch.float32, f"Metadata should have dtype float32 for {sensor['name']}"
        
        print(f"Metadata for {sensor['name']}: {item['metadata']}\n")