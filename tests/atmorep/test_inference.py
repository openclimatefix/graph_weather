"""
Tests for inference utilities from inference.py.

This module verifies:
  - load_model correctly loads a model checkpoint.
  - inference produces outputs with expected shapes.
  - batch_inference processes a dummy dataset.
  - create_forecast extends the time dimension in an autoregressive fashion.

The tests use AtmoRepConfig from config.py.
"""

import os
import torch
import pytest
from graph_weather.models.atmorep.config import AtmoRepConfig
from graph_weather.models.atmorep.inference import (
    load_model,
    inference,
    batch_inference,
    create_forecast,
)
from graph_weather.models.atmorep.model.atmorep import AtmoRep


@pytest.fixture
def config():
    """
    Fixture for an AtmoRep configuration for inference tests.

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
    Fixture for creating sample input data for inference tests.

    Returns:
        dict: Dictionary with each field tensor of shape (B, T, H, W).
    """
    B, T, H, W = 1, config.time_steps, config.spatial_dims[0], config.spatial_dims[1]
    return {field: torch.randn(B, T, H, W) for field in config.input_fields}


def test_load_model(tmp_path, config):
    """
    Test that load_model correctly loads a model checkpoint.

    A dummy checkpoint is saved and then loaded to verify that the model and config are returned.
    """
    model = AtmoRep(config)
    checkpoint = {
        "model": model.state_dict(),
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
    }
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    loaded_model, loaded_config = load_model(str(checkpoint_path), config=config, device="cpu")
    assert isinstance(loaded_model, AtmoRep)
    assert loaded_config.input_fields == config.input_fields


def test_inference_function(config, sample_data):
    """
    Test that the inference function returns predictions with correct shapes.

    Args:
        config: The AtmoRepConfig instance.
        sample_data: Sample input tensors.
    """
    model = AtmoRep(config)
    predictions = inference(model, sample_data)
    for field in config.input_fields:
        assert predictions[field].shape == sample_data[field].shape


def test_batch_inference(config, sample_data):
    """
    Test the batch_inference function on a dummy dataset.

    The dummy dataset is a list of identical sample_data items.
    """
    dataset = [sample_data for _ in range(4)]
    model = AtmoRep(config)
    preds = batch_inference(model, dataset, batch_size=2, num_workers=0)
    for field in config.input_fields:
        B, T, H, W = sample_data[field].shape
        assert preds[field].shape == (4, T, H, W)


def test_create_forecast(config, sample_data):
    """
    Test that create_forecast extends the time dimension appropriately.

    Args:
        config: The AtmoRepConfig instance.
        sample_data: Initial input data.
    """
    model = AtmoRep(config)
    steps = 2
    forecast = create_forecast(model, sample_data, steps=steps)
    for field in config.input_fields:
        assert forecast[field].shape[1] == sample_data[field].shape[1] + steps
