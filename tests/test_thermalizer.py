"""Unit tests for the ThermalizerLayer module."""

import torch

from graph_weather.models.layers.thermalizer import ThermalizerLayer


def test_thermalizer_forward_shape():
    """Test forward pass shape with explicit height, width, and batch."""
    batch_size = 2
    height, width = 12, 12
    nodes = height * width
    features = 256

    x = torch.randn(batch_size * nodes, features)
    layer = ThermalizerLayer(input_dim=features)
    t = torch.randint(0, 1000, (1,)).item()

    out = layer(x, t, height=height, width=width, batch=batch_size)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert torch.isfinite(out).all()


def test_thermalizer_auto_inference():
    """Test forward pass with auto-inferred dimensions (no height, width, batch)."""
    batch_size = 1
    height, width = 12, 12
    nodes = height * width
    features = 256

    x = torch.randn(batch_size * nodes, features)
    layer = ThermalizerLayer(input_dim=features)
    t = torch.randint(0, 1000, (1,)).item()

    out = layer(x, t)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert torch.isfinite(out).all()


def test_thermalizer_different_sizes():
    """Test multiple input grid sizes and batch counts."""
    test_cases = [
        (1, 4, 2, 2),
        (1, 9, 3, 3),
        (1, 16, 4, 4),
        (2, 25, 5, 5),
        (3, 64, 8, 8),
    ]

    features = 256
    layer = ThermalizerLayer(input_dim=features)

    for batch_size, nodes, height, width in test_cases:
        x = torch.randn(batch_size * nodes, features)
        t = torch.randint(0, 1000, (1,)).item()

        out = layer(x, t, height=height, width=width, batch=batch_size)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

        if batch_size == 1:
            out_auto = layer(x, t)
            assert out_auto.shape == x.shape
            assert not torch.isnan(out_auto).any()


def test_grid_reconstruction():
    """Test reshaping from flat to grid format after forward pass."""
    batch_size = 1
    height, width = 6, 8
    nodes = height * width
    features = 256

    x_grid = torch.randn(batch_size, features, height, width)
    x_flat = x_grid.permute(0, 2, 3, 1).reshape(batch_size * nodes, features)

    layer = ThermalizerLayer(input_dim=features)
    t = 100

    out_flat = layer(x_flat, t, height=height, width=width, batch=batch_size)
    out_grid = out_flat.reshape(batch_size, height, width, features).permute(0, 3, 1, 2)

    assert out_flat.shape == x_flat.shape
    assert out_grid.shape == x_grid.shape
    assert not torch.isnan(out_grid).any()
