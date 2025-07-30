import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader
from unittest.mock import patch
from graph_weather.data import AnemoiDataset


def fake_open_dataset(config):
    # Create a small, synthetic xarray.Dataset for testing
    data = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.random.rand(3, 2, 2)),
            "geopotential": (("time", "lat", "lon"), np.random.rand(3, 2, 2)),
            "u_component_of_wind": (("time", "lat", "lon"), np.random.rand(3, 2, 2)),
            "v_component_of_wind": (("time", "lat", "lon"), np.random.rand(3, 2, 2)),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=3),
            "lat": [0.0, 1.0],
            "lon": [10.0, 11.0],
        },
    )
    return data


def test_anemoi_dataset():
    """Test the AnemoiDataset class with synthetic data"""
    dataset_config = {
        "dataset_name": "synthetic",
        "features": [
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "time_range": ("2020-01-01", "2020-01-03"),
        "time_step": 1,
        "max_samples": 2,
        "means": {
            "temperature": 0.5,
            "geopotential": 0.5,
            "u_component_of_wind": 0.5,
            "v_component_of_wind": 0.5,
        },
        "stds": {
            "temperature": 0.2,
            "geopotential": 0.2,
            "u_component_of_wind": 0.2,
            "v_component_of_wind": 0.2,
        },
    }
    with patch("graph_weather.data.anemoi_dataloader.open_dataset", new=fake_open_dataset):
        dataset = AnemoiDataset(**dataset_config)
        assert len(dataset) > 0
        assert dataset.num_lat > 0 and dataset.num_lon > 0
        assert len(dataset.features) > 0

        # Test getting a single sample
        input_data, target_data = dataset[0]
        assert input_data.shape == target_data.shape
        assert input_data.dtype == np.float32
        assert not (
            np.isnan(input_data).any() or np.isnan(target_data).any()
        ), "Found NaN values in data!"

        # Test with DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_input, batch_target = next(iter(dataloader))
        assert batch_input.shape[0] == 2
        assert batch_target.shape[0] == 2


def test_normalization():
    """Test that normalization is working correctly"""
    with patch("graph_weather.data.anemoi_dataloader.open_dataset", new=fake_open_dataset):
        dataset = AnemoiDataset(
            dataset_name="synthetic",
            features=["temperature"],
            max_samples=3,
            means={"temperature": 0.5},
            stds={"temperature": 0.2},
        )
        samples = []
        for i in range(min(3, len(dataset))):
            input_data, _ = dataset[i]
            samples.append(input_data[:, 0])  # First feature (temperature)
        all_values = np.concatenate(samples)
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        assert (
            abs(mean_val) < 0.5 and abs(std_val - 1.0) < 0.5
        ), "Normalization might need adjustment"


def test_time_features():
    """Test that time features are being added correctly"""
    with patch("graph_weather.data.anemoi_dataloader.open_dataset", new=fake_open_dataset):
        dataset = AnemoiDataset(
            dataset_name="synthetic",
            features=["temperature"],
            max_samples=2,
            means={"temperature": 0.5},
            stds={"temperature": 0.2},
        )
        input_data, _ = dataset[0]
        num_features = len(dataset.features)
        num_time_features = 4  # sin/cos day, sin/cos hour
        expected_total_features = num_features + num_time_features
        actual_features = input_data.shape[1]
        assert actual_features == expected_total_features, "Feature count mismatch"


def test_check_anemoi_dataset_output():
    """Compare output format with GenCast dataloader expectations"""
    with patch("graph_weather.data.anemoi_dataloader.open_dataset", new=fake_open_dataset):
        dataset = AnemoiDataset(
            dataset_name="synthetic",
            features=["temperature", "geopotential"],
            max_samples=2,
            means={"temperature": 0.5, "geopotential": 0.5},
            stds={"temperature": 0.2, "geopotential": 0.2},
        )
        input_data, target_data = dataset[0]
        assert input_data.shape == target_data.shape, "Input and target shapes do not match"
        assert input_data.dtype == np.float32, "Input data type should be float32"
        assert len(input_data.shape) == 2, "Input data should be 2D"
        expected_locations = dataset.num_lat * dataset.num_lon
        actual_locations = input_data.shape[0]
        assert actual_locations == expected_locations, "Format mismatch"
