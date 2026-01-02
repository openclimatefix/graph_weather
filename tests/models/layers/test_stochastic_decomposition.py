"""Tests for the Stochastic Decomposition Layer."""

import pytest
import torch

from graph_weather.models.layers.stochastic_decomposition import (
    StochasticDecompositionLayer,
)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 32, 10),
        (2, 32, 16, 16),
        (2, 32, 8, 16, 16),
    ],
)
def test_sdl_shapes(shape):
    """Ensure SDL handles arbitrary spatial/temporal dimensions via broadcasting"""
    batch, channels = shape[0], shape[1]
    latent_dim = 16

    x = torch.randn(*shape)
    z = torch.randn(batch, latent_dim)

    model = StochasticDecompositionLayer(input_dim=channels, latent_dim=latent_dim)
    out = model(x, z)

    assert out.shape == shape
    assert not torch.isnan(out).any()


def test_initialization_is_deterministic():
    """Alpha initialized to 0 should imply Identity function initially"""
    x = torch.randn(2, 64, 32, 32)
    z = torch.randn(2, 16)

    model = StochasticDecompositionLayer(input_dim=64, latent_dim=16)

    assert torch.allclose(model.alpha, torch.zeros_like(model.alpha))

    out = model(x, z)
    assert torch.allclose(out, x, atol=1e-6)


def test_reproducibility():
    """Fixed seed + fixed latent = fixed output"""
    x = torch.randn(2, 16, 10)
    z = torch.randn(2, 8)

    model = StochasticDecompositionLayer(16, 8)

    with torch.no_grad():
        model.alpha.fill_(0.5)

    torch.manual_seed(42)
    out1 = model(x, z)

    torch.manual_seed(42)
    out2 = model(x, z)

    assert torch.equal(out1, out2)


def test_gradient_flow():
    """Test that gradients flow correctly through the layer."""
    x = torch.randn(2, 16, 10, requires_grad=True)
    z = torch.randn(2, 8, requires_grad=True)

    model = StochasticDecompositionLayer(16, 8)
    with torch.no_grad():
        model.alpha.fill_(0.1)

    out = model(x, z)
    loss = out.sum()
    loss.backward()

    assert model.style_net.weight.grad is not None
    assert model.alpha.grad is not None
    assert x.grad is not None


def test_channel_mismatch_error():
    """Test that channel mismatch raises ValueError."""
    x = torch.randn(2, 32, 10)
    z = torch.randn(2, 8)
    model = StochasticDecompositionLayer(input_dim=16, latent_dim=8)

    with pytest.raises(ValueError):
        model(x, z)
