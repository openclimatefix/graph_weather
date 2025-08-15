"""Unit tests for NNJA-AI data loading components.

Tests cover variable classification, dataset loading, and PyTorch integration.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from graph_weather.data.nnja_ai import (
    NNJAXarrayAsTorchDataset,
    SensorDataset,
    _classify_variable,
    load_nnja_dataset,
)


@pytest.fixture
def mock_datacatalog():
    """Fixture to mock the DataCatalog with properly configured variables."""
    with patch("graph_weather.data.nnja_ai.DataCatalog") as mock:
        mock_catalog = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.load_manifest = MagicMock()

        # Setup valid variables with proper attributes
        def create_mock_var(var_type):
            var = MagicMock()
            var.category = var_type
            return var

        valid_variables = {
            "time": create_mock_var("primary_descriptor"),
            "latitude": create_mock_var("primary_descriptor"),
            "longitude": create_mock_var("primary_descriptor"),
            "TMBR_00001": create_mock_var("primary_data"),
            "TMBR_00002": create_mock_var("primary_data"),
            "OBS_TIMESTAMP": create_mock_var("primary_descriptor"),
            "LAT": create_mock_var("primary_descriptor"),
            "LON": create_mock_var("primary_descriptor"),
        }

        def mock_sel(time=None, variables=None):
            # Always include time plus requested variables
            vars_to_load = ["time"]
            if variables:
                for v in variables:
                    if v == "LAT":
                        vars_to_load.append("latitude")
                    elif v == "LON":
                        vars_to_load.append("longitude")
                    else:
                        vars_to_load.append(v)

            # Validate variables exist
            invalid_vars = [v for v in vars_to_load if v not in valid_variables]
            if invalid_vars:
                raise ValueError(f"Invalid variables requested: {invalid_vars}")
            return mock_dataset

        mock_dataset.sel = mock_sel
        mock_dataset.variables = valid_variables

        def mock_load_dataset(backend="pandas", engine="pyarrow"):
            time_points = pd.date_range(
                start=datetime(2021, 1, 1), periods=100, freq="h"
            )

            data = {
                "time": time_points,
                "latitude": np.full(100, 45.0),
                "longitude": np.full(100, -120.0),
                "TMBR_00001": np.full(100, 250.0),
                "TMBR_00002": np.full(100, 250.0),
            }
            return pd.DataFrame(data)

        mock_dataset.load_dataset = mock_load_dataset
        mock_catalog.__getitem__.side_effect = lambda name: mock_dataset
        mock.return_value = mock_catalog

        yield mock


def test_variable_classification():
    """Test the variable classification logic."""
    # Create a mock variable with category attribute
    mock_var = MagicMock()
    mock_var.category = "primary_data"
    assert _classify_variable(mock_var) == "primary_data"


def test_load_nnja_dataset(mock_datacatalog):
    """Test the core dataset loading function."""
    ds = load_nnja_dataset("test-dataset", time=datetime(2021, 1, 1))

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.dims
    assert len(ds.data_vars) >= 3
    assert np.issubdtype(ds.time.dtype, np.datetime64)


def test_sensor_dataset(mock_datacatalog):
    """Test the SensorDataset class."""
    ds = SensorDataset(
        "test-dataset",
        time=datetime(2021, 1, 1),
        variables=["LAT", "LON", "TMBR_00001"],  # Used original names here
    )

    assert len(ds) == 100
    sample = ds[0]
    assert isinstance(sample, dict)
    assert "latitude" in sample or "LAT" in sample
    assert "longitude" in sample or "LON" in sample
    assert "TMBR_00001" in sample


def test_nnja_xarray_torch_dataset(mock_datacatalog):
    """Test the xarray to torch Dataset adapter."""
    xrds = load_nnja_dataset("test-dataset")
    torch_ds = NNJAXarrayAsTorchDataset(xrds)

    assert len(torch_ds) == len(xrds.time)
    sample = torch_ds[0]
    assert isinstance(sample, dict)


def test_custom_variable_selection(mock_datacatalog):
    """Verify loading specific variables works with coordinate renaming."""
    # Request the original coordinate names
    # Request original coordinate names
    custom_vars = ["LAT", "LON"]
    ds = load_nnja_dataset("test-dataset", variables=custom_vars)

    # Check renamed coordinates exist
    assert "latitude" in ds.data_vars
    assert "longitude" in ds.data_vars


def test_load_all_variables(mock_datacatalog):
    """Test loading all variables."""
    ds = load_nnja_dataset("test-dataset", load_all=True)
    assert len(ds.data_vars) >= 4  # time + lat + lon + at least one variable
