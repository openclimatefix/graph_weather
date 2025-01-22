"""
Tests for the nnjai_wrapp module in the graph_weather package.

This file contains unit tests for AMSUDataset and collate_fn functions.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from graph_weather.data.nnjai_wrapp import AMSUDataset, collate_fn


# Mock the DataCatalog to avoid actual data loading
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


def test_amsu_dataset(mock_datacatalog):
    """
    Test the AMSUDataset class to ensure proper data loading and tensor structure.

    This test validates the AMSUDataset class for its ability to load the dataset
    correctly, check for the appropriate tensor properties, and ensure the keys
    and data types match expectations.
    """
    # Initialize dataset parameters
    dataset_name = "amsua-1bamua-NC021023"
    time = "2021-01-01 00Z"
    primary_descriptors = ["OBS_TIMESTAMP", "LAT", "LON"]
    additional_variables = ["TMBR_00001", "TMBR_00002"]

    dataset = AMSUDataset(dataset_name, time, primary_descriptors, additional_variables)

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
    assert item["metadata"].shape == (
        len(additional_variables),
    ), f"Metadata shape mismatch. Expected ({len(additional_variables)},)."
    assert item["metadata"].dtype == torch.float32, "Metadata should have dtype float32."


def test_collate_function():
    """
    Test the collate_fn function to ensure proper batching of dataset items.

    This test checks that the collate_fn properly batches the timestamp, latitude,
    longitude, and metadata fields of the dataset, ensuring correct shapes and data types.
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
    assert batched["metadata"].shape == (
        batch_size,
        metadata_size,
    ), "Metadata batch shape mismatch."

    assert batched["timestamp"].dtype == torch.float32, "Timestamp dtype mismatch."
    assert batched["latitude"].dtype == torch.float32, "Latitude dtype mismatch."
    assert batched["longitude"].dtype == torch.float32, "Longitude dtype mismatch."
    assert batched["metadata"].dtype == torch.float32, "Metadata dtype mismatch."
