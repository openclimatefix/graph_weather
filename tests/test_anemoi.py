import sys
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

# Add the project root to Python path so we can import from graph_weather
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_weather.data import AnemoiDataset


def test_anemoi_dataset():
    """Test the AnemoiDataset class with real or fixture data"""
    dataset_config = {
        "dataset_name": "era5-test",
        "features": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "time_range": ("2020-01-01", "2020-01-31"),
        "time_step": 1,
        "max_samples": 10,
    }
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
    dataset = AnemoiDataset(dataset_name="test", features=["temperature"], max_samples=5)
    samples = []
    for i in range(min(3, len(dataset))):
        input_data, _ = dataset[i]
        samples.append(input_data[:, 0])  # First feature (temperature)
    all_values = np.concatenate(samples)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    assert abs(mean_val) < 0.5 and abs(std_val - 1.0) < 0.5, "Normalization might need adjustment"


def test_time_features():
    """Test that time features are being added correctly"""
    dataset = AnemoiDataset(dataset_name="test", features=["temperature"], max_samples=2)
    input_data, _ = dataset[0]
    num_features = len(dataset.features)
    num_time_features = 4  # sin/cos day, sin/cos hour
    expected_total_features = num_features + num_time_features
    actual_features = input_data.shape[1]
    assert actual_features == expected_total_features, "Feature count mismatch"


def test_compare_with_gencast_format():
    """Compare output format with GenCast dataloader expectations"""
    dataset = AnemoiDataset(
        dataset_name="test", features=["temperature", "geopotential"], max_samples=2
    )
    input_data, target_data = dataset[0]
    assert input_data.shape == target_data.shape, "Input and target shapes do not match"
    assert input_data.dtype == np.float32, "Input data type should be float32"
    assert len(input_data.shape) == 2, "Input data should be 2D"
    expected_locations = dataset.num_lat * dataset.num_lon
    actual_locations = input_data.shape[0]
    assert actual_locations == expected_locations, "Format mismatch"
