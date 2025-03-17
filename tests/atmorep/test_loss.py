"""
Tests for the AtmoRepLoss module.

This module tests the loss computations from loss.py by verifying:
  - The reconstruction and physical consistency components are computed.
  - Loss values are non-negative with and without masks.

The tests use AtmoRepConfig from config.py.
"""

import pytest
import torch
from graph_weather.models.atmorep.training.loss import AtmoRepLoss
from graph_weather.models.atmorep.config import AtmoRepConfig


@pytest.fixture
def config():
    """
    Fixture for an AtmoRep configuration for loss tests.

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
def dummy_data(config):
    """
    Fixture to generate dummy target data.

    Returns:
        dict: Dictionary of tensors for each input field of shape (B, T, H, W).
    """
    B, T, H, W = 1, config.time_steps, config.spatial_dims[0], config.spatial_dims[1]
    return {field: torch.randn(B, T, H, W) for field in config.input_fields}


@pytest.fixture
def dummy_predictions(config):
    """
    Fixture to generate dummy predictions with an ensemble dimension.

    Returns:
        dict: Dictionary of tensors with shape (E, B, T, H, W) for each field.
    """
    B, T, H, W = 1, config.time_steps, config.spatial_dims[0], config.spatial_dims[1]
    E = 2
    return {field: torch.randn(E, B, T, H, W) for field in config.input_fields}


@pytest.fixture
def dummy_masks(config):
    """
    Fixture to generate dummy masks (all ones).

    Returns:
        dict: Dictionary of mask tensors with shape (B, T, H, W) for each field.
    """
    B, T, H, W = 1, config.time_steps, config.spatial_dims[0], config.spatial_dims[1]
    return {field: torch.ones(B, T, H, W) for field in config.input_fields}


def test_loss_computation_with_masks(config, dummy_predictions, dummy_data, dummy_masks):
    """
    Test that AtmoRepLoss computes a valid loss when masks are provided.

    Args:
        config: The AtmoRepConfig instance.
        dummy_predictions: Dummy prediction tensors.
        dummy_data: Dummy target tensors.
        dummy_masks: Dummy mask tensors.
    """
    loss_fn = AtmoRepLoss(config.input_fields, recon_weight=1.0, phys_weight=0.1)
    total_loss, loss_details = loss_fn(dummy_predictions, dummy_data, dummy_masks)
    assert isinstance(total_loss, torch.Tensor)
    assert "reconstruction" in loss_details
    assert "physical" in loss_details
    assert "field_losses" in loss_details
    assert total_loss.item() >= 0


def test_loss_computation_without_masks(config, dummy_predictions, dummy_data):
    """
    Test that AtmoRepLoss computes a valid loss when no masks are provided.

    Args:
        config: The AtmoRepConfig instance.
        dummy_predictions: Dummy prediction tensors.
        dummy_data: Dummy target tensors.
    """
    loss_fn = AtmoRepLoss(config.input_fields, recon_weight=1.0, phys_weight=0.1)
    total_loss, loss_details = loss_fn(dummy_predictions, dummy_data)
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() >= 0
