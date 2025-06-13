"""
Tests for the AtmoRep model.

This module tests key functionalities of the AtmoRep model (from atmorep.py), including:
  - Basic forward pass to verify output shapes.
  - Ensemble predictions that add an ensemble dimension.
  - Autoregressive forecasting via the create_forecast helper.

The tests use a standard AtmoRepConfig instance from config.py.
"""

import pytest
import torch
from graph_weather.models.atmorep.model.atmorep import AtmoRep
from graph_weather.models.atmorep.inference import create_forecast
from graph_weather.models.atmorep.config import AtmoRepConfig


@pytest.fixture
def config():
    """
    Fixture for an AtmoRep configuration using AtmoRepConfig from config.py.

    Returns:
        AtmoRepConfig: A configuration object with testing parameters.
    """
    return AtmoRepConfig(
        input_fields=["t2m", "u10"],
        spatial_dims=(32, 32),
        patch_size=8,
        time_steps=4,
        mask_ratio=0.5,
        model_dims={"encoder": 64, "decoder": 64, "projection": 256, "embedding": 512},
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=0.1,
        attention_dropout=0.1,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        epochs=5,
        warmup_epochs=1,
        year_month_samples=4,
        time_slices_per_ym=6,
        neighborhoods_per_slice=(2, 8),
        neighborhood_size=(32, 32),
        num_ensemble_members=3,
    )


@pytest.fixture
def sample_data(config):
    """
    Fixture to create sample input data for the model.

    Args:
        config (AtmoRepConfig): The configuration object.

    Returns:
        dict: Dictionary mapping each input field to a tensor of shape (B, T, H, W).
    """
    B, T, H, W = 2, config.time_steps, config.spatial_dims[0], config.spatial_dims[1]
    return {field: torch.randn(B, T, H, W) for field in config.input_fields}


def test_forward_pass(config, sample_data):
    """
    Test that a forward pass produces outputs with the expected shapes.

    Args:
        config: The AtmoRepConfig instance.
        sample_data: Input data for the model.
    """
    model = AtmoRep(config)
    outputs = model(sample_data)
    for field in config.input_fields:
        assert outputs[field].shape == sample_data[field].shape


def test_ensemble_predictions(config, sample_data):
    """
    Test that ensemble predictions include an ensemble dimension.

    Args:
        config: The AtmoRepConfig instance.
        sample_data: Input data for the model.
    """
    model = AtmoRep(config)
    ensemble_size = config.num_ensemble_members
    outputs = model(sample_data, ensemble_size=ensemble_size)
    for field in config.input_fields:
        out = outputs[field]
        assert out.dim() == 5
        assert out.shape[0] == ensemble_size
        assert out.shape[1:] == sample_data[field].shape


def test_autoregressive_forecast(config, sample_data):
    """
    Test that create_forecast extends the time dimension in an autoregressive manner.

    Args:
        config: The AtmoRepConfig instance.
        sample_data: Initial input data.
    """
    model = AtmoRep(config)
    steps = 2
    forecast = create_forecast(model, sample_data, steps=steps)
    for field in config.input_fields:
        assert forecast[field].shape[1] == sample_data[field].shape[1] + steps
