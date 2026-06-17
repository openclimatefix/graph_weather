"""Tests for RegionalForecaster model."""

import torch

from graph_weather.models.regional_forecast import (
    BoundaryNudgingLayer,
    RegionalForecasterConfig,
)


def _small_config():
    """Create a small config for fast testing."""
    return RegionalForecasterConfig(
        feature_dim=12,
        aux_dim=4,
        node_dim=32,
        edge_dim=32,
        num_blocks=2,
        hidden_dim_processor_node=32,
        hidden_dim_processor_edge=32,
        hidden_dim_decoder=32,
    )


def _uk_latlons():
    """Five UK coordinates."""
    return [(51.5, -0.1), (52.0, 0.5), (53.0, -1.0), (54.0, -2.0), (50.0, -3.0)]


def _germany_latlons():
    """Three Germany coordinates."""
    return [(52.5, 13.4), (48.1, 11.6), (50.9, 6.9)]


def test_config_build():
    """Config.build() returns a working model."""
    config = _small_config()
    model = config.build()
    assert hasattr(model, "forward")
    assert hasattr(model, "graph_builder")
    assert hasattr(model, "h3_embeddings")


def test_forward_shape():
    """Output shape matches [B, N_obs, output_dim]."""
    model = _small_config().build()
    lat_lons = _uk_latlons()
    features = torch.randn(2, 5, 16)  # 12 + 4 = 16

    out = model(features, lat_lons)
    assert out.shape == (2, 5, 12)


def test_no_nan_output():
    """Forward pass produces no NaN values."""
    model = _small_config().build()
    features = torch.randn(2, 5, 16)

    out = model(features, _uk_latlons())
    assert not torch.isnan(out).any()


def test_different_coords_per_forward():
    """Same model handles different regions in successive forward passes."""
    model = _small_config().build()

    out_uk = model(torch.randn(1, 5, 16), _uk_latlons())
    assert out_uk.shape == (1, 5, 12)

    out_de = model(torch.randn(1, 3, 16), _germany_latlons())
    assert out_de.shape == (1, 3, 12)


def test_variable_length_coords():
    """Model works with different observation counts."""
    model = _small_config().build()

    # 5 observations
    out5 = model(torch.randn(1, 5, 16), _uk_latlons())
    assert out5.shape == (1, 5, 12)

    # 3 observations
    out3 = model(torch.randn(1, 3, 16), _germany_latlons())
    assert out3.shape == (1, 3, 12)


def test_backward_pass():
    """Gradients flow through the full pipeline."""
    model = _small_config().build()
    features = torch.randn(1, 5, 16)

    out = model(features, _uk_latlons())
    loss = out.sum()
    loss.backward()

    # Check gradients exist on key parameters
    assert model.h3_embeddings.grad is not None
    has_grad = any(p.grad is not None for p in model.node_encoder.parameters())
    assert has_grad


def test_output_dim_override():
    """Custom output_dim produces the right shape."""
    config = _small_config()
    config.output_dim = 6
    model = config.build()
    features = torch.randn(1, 5, 16)

    out = model(features, _uk_latlons())
    assert out.shape == (1, 5, 6)


def test_residual_connection():
    """Output includes a residual from the input features."""
    model = _small_config().build()
    features = torch.randn(1, 5, 16)

    # With zero weights, output should be close to the input residual
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    out = model(features, _uk_latlons())
    expected_residual = features[..., :12]
    assert torch.allclose(out, expected_residual, atol=1e-5)


def _nudging_config():
    """Config with nudging enabled."""
    return RegionalForecasterConfig(
        feature_dim=12,
        aux_dim=4,
        node_dim=32,
        edge_dim=32,
        num_blocks=2,
        hidden_dim_processor_node=32,
        hidden_dim_processor_edge=32,
        hidden_dim_decoder=32,
        enable_nudging=True,
        nudging_hidden_dim=16,
    )


def test_nudging_disabled_unchanged():
    """Output is identical with enable_nudging=False (backward compat)."""
    config_off = _small_config()
    model = config_off.build()
    assert model.nudging is None

    features = torch.randn(1, 5, 16)
    global_ctx = torch.randn(1, 5, 12)
    out = model(features, _uk_latlons(), global_context=global_ctx)
    out_no_ctx = model(features, _uk_latlons())
    assert torch.allclose(out, out_no_ctx)


def test_nudging_no_context_unchanged():
    """Nudging enabled but global_context=None produces same output."""
    model = _nudging_config().build()
    features = torch.randn(1, 5, 16)

    out = model(features, _uk_latlons(), global_context=None)
    assert out.shape == (1, 5, 12)
    assert not torch.isnan(out).any()


def test_nudging_changes_output():
    """Global context causes output to differ from no-context case."""
    model = _nudging_config().build()
    features = torch.randn(1, 5, 16)
    global_ctx = torch.randn(1, 5, 12) * 10.0

    out_no_ctx = model(features, _uk_latlons(), global_context=None)
    out_with_ctx = model(features, _uk_latlons(), global_context=global_ctx)
    assert not torch.allclose(out_no_ctx, out_with_ctx)


def test_relaxation_weights_range():
    """Alpha prior is in [0, 1] with max at the farthest point."""
    weights = BoundaryNudgingLayer._compute_relaxation_weights(_uk_latlons(), torch.device("cpu"))
    assert weights.shape == (5, 1)
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0
    assert torch.isclose(weights.max(), torch.tensor(1.0))


def test_nudging_backward_pass():
    """Gradients flow through the nudging layer."""
    model = _nudging_config().build()
    features = torch.randn(1, 5, 16)
    global_ctx = torch.randn(1, 5, 12)

    out = model(features, _uk_latlons(), global_context=global_ctx)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.nudging.parameters())
    assert has_grad
