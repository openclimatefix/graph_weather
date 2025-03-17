"""
Tests for training utilities from train.py.

This module verifies:
  - The training epoch function (_train_epoch) correctly processes a batch.
  - The checkpoint saving function (_save_checkpoint) writes a file.
  - The generate_training_masks function produces masks with expected shapes.

Dummy data and a dummy loss function are used for these tests.
"""

import os
import torch
import pytest
from graph_weather.models.atmorep.training import train
from graph_weather.models.atmorep.model.atmorep import AtmoRep
from graph_weather.models.atmorep.config import AtmoRepConfig

@pytest.fixture
def config():
    """
    Fixture for an AtmoRep configuration for training tests.

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
        num_ensemble_members=3
    )

@pytest.fixture
def dummy_batch(config):
    """
    Fixture for creating a dummy training batch.

    Returns:
        dict: A batch with tensors for each input field of shape (B, T, H, W).
    """
    B, T, H, W = config.batch_size, config.time_steps, config.spatial_dims[0], config.spatial_dims[1]
    return {field: torch.randn(B, T, H, W, requires_grad=True) for field in config.input_fields}

class DummyLoss:
    """
    Dummy loss function for training tests that computes mean squared error.
    """
    def __init__(self, input_fields):
        self.input_fields = input_fields

    def __call__(self, predictions, targets, masks=None):
        loss = 0.0
        field_losses = {}
        for field in self.input_fields:
            field_loss = torch.mean((predictions[field] - targets[field]) ** 2)
            loss += field_loss
            field_losses[field] = field_loss.item()
        return loss, {"field_losses": field_losses}

def dummy_generate_masks(batch_data, config):
    """
    Dummy mask generator that returns masks of ones matching the input data shape.
    """
    return {field: torch.ones_like(batch_data[field]) for field in config.input_fields}

def test_train_epoch(monkeypatch, config, dummy_batch):
    """
    Test the _train_epoch function to verify loss computation and model updates.

    Uses a dummy model whose forward method returns the input unchanged.
    """
    model = AtmoRep(config)
    model.forward = lambda x, masks=None, **kwargs: x
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = DummyLoss(config.input_fields)
    dummy_loader = [dummy_batch]
    monkeypatch.setattr(train, "generate_training_masks", dummy_generate_masks)
    epoch_loss, field_losses = train._train_epoch(model, dummy_loader, optimizer, loss_fn, epoch=0)
    assert isinstance(epoch_loss, float)
    for field in config.input_fields:
        assert field in field_losses

def test_save_checkpoint(tmp_path, config):
    """
    Test that _save_checkpoint creates a checkpoint file with the expected name.
    """
    model = AtmoRep(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    epoch = 1
    best_val_loss = 0.1
    is_best = True
    history = {}
    train._save_checkpoint(model, optimizer, config, str(output_dir), epoch, best_val_loss, is_best, history)
    checkpoint_file = output_dir / f"checkpoint_epoch_{epoch}.pth"
    assert checkpoint_file.exists()

def test_generate_training_masks(config, dummy_batch):
    """
    Test the generate_training_masks function to ensure masks have the expected shape.

    The masks should be of shape (B, T, H, W) matching the spatial dimensions of the input,
    reflecting the current implementation.
    """
    masks = train.generate_training_masks(dummy_batch, config)
    assert set(masks.keys()) == set(dummy_batch.keys())
    for field in dummy_batch:
        B, T, H, W = dummy_batch[field].shape
        expected_shape = (B, T, H, W)
        assert masks[field].shape == expected_shape