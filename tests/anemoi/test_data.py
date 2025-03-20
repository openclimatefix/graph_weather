import numpy as np
import xarray as xr
import pandas as pd
import pytest
import torch

from graph_weather.data import anemoi_dataloader

# Import the class from the module
AnemoiDataset = anemoi_dataloader.AnemoiDataset


def create_dummy_data():
    """
    Create a dummy xarray Dataset with two features ('temperature' and 'geopotential'),
    and coordinates for time, latitude, and longitude.
    """
    times = pd.date_range("2016-01-01", periods=10, freq="D")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 10)

    # Create dummy data for two features.
    temp = np.random.rand(10, 10, 10).astype(np.float32) * 300
    geopot = np.random.rand(10, 10, 10).astype(np.float32) * 5000

    ds = xr.Dataset(
        {
            "temperature": (["time", "latitude", "longitude"], temp),
            "geopotential": (["time", "latitude", "longitude"], geopot),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    return ds


@pytest.fixture(scope="session")
def dummy_zarr_path(tmp_path_factory):
    """
    Create a temporary Zarr store on disk containing dummy data for testing.
    This avoids keeping large datasets in memory, improving scalability.
    """
    ds = create_dummy_data()
    zarr_dir = tmp_path_factory.mktemp("data") / "dummy.zarr"
    ds.to_zarr(zarr_dir, mode="w")
    return str(zarr_dir)


@pytest.fixture(autouse=True)
def patch_open_dataset(monkeypatch):
    """
    Patch the open_dataset function in the anemoi_dataloader module so that it
    calls xr.open_zarr. This ensures our tests load the Zarr store from disk.
    """
    monkeypatch.setattr(anemoi_dataloader, "open_dataset", xr.open_zarr)


def test_dataset_length(dummy_zarr_path):
    """
    Test that the dataset length equals the number of time steps.
    """
    features = ["temperature", "geopotential"]
    dataset = AnemoiDataset(
        filepath=dummy_zarr_path, features=features, start_year=2016, end_year=2016
    )
    # Our dummy dataset has 10 time steps.
    assert len(dataset) == 10


def test_getitem_output_shape(dummy_zarr_path):
    """
    Test that __getitem__ returns a tensor with the expected shape.
    The expected channel dimension is the number of meteorological features plus 5 geographical channels.
    """
    features = ["temperature", "geopotential"]
    dataset = AnemoiDataset(
        filepath=dummy_zarr_path, features=features, start_year=2016, end_year=2016
    )
    sample = dataset[0]  # Get the first sample.
    expected_channels = len(features) + 5  # meteorological + 4 geo channels + 1 day-of-year channel
    # Our dummy dataset has 10 latitudes and 10 longitudes.
    assert sample.shape == (expected_channels, 10, 10)


def test_normalization(dummy_zarr_path):
    """
    Verify that the meteorological features (first channels) are normalized between 0 and 1.
    """
    features = ["temperature", "geopotential"]
    dataset = AnemoiDataset(
        filepath=dummy_zarr_path, features=features, start_year=2016, end_year=2016
    )
    sample = dataset[0].numpy()
    meteo_features = sample[: len(features)]
    # Check that all meteorological feature values are between 0 and 1.
    assert np.all(meteo_features >= 0)
    assert np.all(meteo_features <= 1)


def test_day_of_year(dummy_zarr_path):
    """
    Test that the dynamic day-of-year feature (last geographical channel) is correctly computed.
    For the first sample ("2016-01-01"), the normalized day-of-year should be 1/366.
    """
    features = ["temperature", "geopotential"]
    dataset = AnemoiDataset(
        filepath=dummy_zarr_path, features=features, start_year=2016, end_year=2016
    )
    sample = dataset[0].numpy()
    expected_channels = len(features) + 5
    doy_channel = sample[expected_channels - 1]
    # Ensure the day-of-year feature is constant across the spatial dimensions.
    assert np.allclose(doy_channel, doy_channel[0, 0])
    expected_doy = 1 / 366
    np.testing.assert_allclose(doy_channel[0, 0], expected_doy, atol=1e-3)


def test_tensor_type(dummy_zarr_path):
    """
    Ensure that __getitem__ returns a PyTorch tensor.
    """
    features = ["temperature", "geopotential"]
    dataset = AnemoiDataset(
        filepath=dummy_zarr_path, features=features, start_year=2016, end_year=2016
    )
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
