"""Tests for the GenDA model."""

import numpy as np
import torch
import pytest

from graph_weather.models.genda.model import GenDA


def test_genda_forward():
    """Test GenDA forward pass on a regular grid."""
    model = GenDA(
        grid_lon=np.linspace(0, 1, 2),
        grid_lat=np.linspace(0, 1, 2),
        input_features_dim=2,
        output_features_dim=2,
        hidden_dims=[16, 16],
        num_blocks=1,
        num_heads=1,
        splits=1,
        num_hops=1,
    )

    corrupted_targets = torch.randn(1, 2, 2, 2)
    prev_inputs = torch.randn(1, 2, 2, 4)
    sensor_mask = torch.ones(1, 2, 2, 1)
    sensor_values = torch.randn(1, 2, 2, 1)

    noise = torch.ones(1, 1)

    out = model(
        corrupted_targets=corrupted_targets,
        prev_inputs=prev_inputs,
        noise_levels=noise,
        sensor_mask=sensor_mask,
        sensor_values=sensor_values,
    )

    assert out.shape == corrupted_targets.shape


def test_genda_irregular_graph():
    """Test GenDA forward pass on irregular graph points."""

    grid_lon = np.array([0.1, 0.7, 0.3, 0.95])
    grid_lat = np.array([0.2, 0.8, 0.4, 0.6])

    model = GenDA(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        input_features_dim=2,
        output_features_dim=2,
        hidden_dims=[16, 16],
        num_blocks=1,
        num_heads=1,
        splits=1,
        num_hops=1,
    )

    batch_size = 1
    num_points = len(grid_lon)

    corrupted_targets = torch.randn(batch_size, num_points, 2)
    noise_levels = torch.randn(batch_size)
    prev_inputs = torch.randn(batch_size, num_points, 2)

    sensor_mask = torch.randint(0, 2, (batch_size, num_points, 1)).float()
    sensor_values = torch.randn(batch_size, num_points, 2)

    with pytest.raises(ValueError):
        model(
            corrupted_targets=corrupted_targets,
            prev_inputs=prev_inputs,
            noise_levels=noise_levels,
            sensor_mask=sensor_mask,
            sensor_values=sensor_values,
        )
