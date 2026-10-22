"""Tests for the multi-grid FGN model."""
import numpy as np
import torch

from graph_weather.models.fgn.multigrid_model import MultiGridFGN


def test_multigrid_fgn_forward_and_backward():
    """Test predictions, output shapes, and gradient propagation across grids."""
    lon_a = np.arange(0, 360, 30)
    lat_a = np.arange(-90, 91, 30)

    lon_b = np.arange(0, 360, 20)
    lat_b = np.arange(-90, 91, 30)

    model = MultiGridFGN(
        grid_lons=[lon_a, lon_b],
        grid_lats=[lat_a, lat_b],
        input_features_dims=[3, 3],
        output_features_dims=[2, 4],
        noise_dimension=4,
        hidden_dims=[8, 16],
        num_blocks=2,
        num_heads=2,
        splits=0,
        num_hops=1,
    )

    input_a = torch.randn(2, len(lon_a), len(lat_a), 3)
    input_b = torch.randn(2, len(lon_b), len(lat_b), 3)

    outputs = model([input_a, input_b], num_ensemble=2)

    assert len(outputs) == 2
    assert outputs[0].shape == (2, 2, len(lon_a), len(lat_a), 2)
    assert outputs[1].shape == (2, 2, len(lon_b), len(lat_b), 4)

    assert all(torch.isfinite(output).all() for output in outputs)

    loss = sum(output.square().mean() for output in outputs)
    loss.backward()

    assert any(p.grad is not None for p in model.encoders[0].parameters())
    assert any(p.grad is not None for p in model.encoders[1].parameters())
    assert any(p.grad is not None for p in model.processor.parameters())


def test_multigrid_fgn_requires_multiple_grids():
    """Reject configurations containing only one grid."""
    lon = np.arange(0, 360, 30)
    lat = np.arange(-90, 91, 30)

    try:
        MultiGridFGN(
            grid_lons=[lon],
            grid_lats=[lat],
            input_features_dims=[3],
            output_features_dims=[2],
            noise_dimension=4,
            hidden_dims=[8, 16],
            num_heads=2,
        )
    except ValueError as exc:
        assert "at least two grids" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a single grid")

def test_multigrid_fgn_shares_information_between_grids():
    """Verify that one grid can influence another through the shared mesh."""
    lon_a = np.arange(0, 360, 30)
    lat_a = np.arange(-90, 91, 30)
    lon_b = np.arange(0, 360, 20)
    lat_b = np.arange(-90, 91, 30)

    torch.manual_seed(42)
    model = MultiGridFGN(
        grid_lons=[lon_a, lon_b],
        grid_lats=[lat_a, lat_b],
        input_features_dims=[3, 3],
        output_features_dims=[2, 4],
        noise_dimension=4,
        hidden_dims=[8, 16],
        num_blocks=2,
        num_heads=2,
        splits=0,
        num_hops=1,
    )
    model.eval()

    input_a = torch.randn(1, len(lon_a), len(lat_a), 3)
    input_b = torch.randn(1, len(lon_b), len(lat_b), 3)

    torch.manual_seed(123)
    output_a = model([input_a, input_b], num_ensemble=1)[0]

    torch.manual_seed(123)
    changed_output_a = model(
        [input_a, input_b + 1.0],
        num_ensemble=1,
    )[0]

    assert not torch.allclose(output_a, changed_output_a)
