"""Integration tests for GraphWeatherForecaster using the ThermalizerLayer."""

import torch

from graph_weather.models.forecast import GraphWeatherForecaster
from graph_weather.models.layers.thermalizer import ThermalizerLayer


def test_gencast_thermal_integration():
    """End-to-end test: GraphWeatherForecaster with ThermalizerLayer on a 3x3 grid."""
    lat_lons = [(i // 3, i % 3) for i in range(9)]  # 3x3 grid

    model = GraphWeatherForecaster(
        lat_lons,
        use_thermalizer=True,
        feature_dim=3,
        aux_dim=0,
        node_dim=256,
        num_blocks=1,
    )

    features = torch.randn(1, len(lat_lons), 3)
    t = torch.randint(0, 1000, (1,)).item()
    pred = model(features, t=t)

    assert not torch.isnan(pred).any()
    assert torch.isfinite(pred).all()

    if pred.dim() == 4:
        assert pred.shape[0] == features.shape[0]
        assert pred.shape[2] * pred.shape[3] == features.shape[1]
    else:
        assert pred.shape == features.shape


def test_thermalizer_small_grids():
    """Test ThermalizerLayer on various small grid sizes."""
    layer = ThermalizerLayer(input_dim=256)
    t = torch.randint(0, 1000, (1,)).item()

    for nodes in [4, 9, 64]:  # 2x2, 3x3, 8x8
        x = torch.randn(nodes, 256)
        out = layer(x, t)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


def test_small_grid_integration():
    """Test GraphWeatherForecaster + ThermalizerLayer on a 2x2 grid."""
    lat_lons = [(i // 2, i % 2) for i in range(4)]  # 2x2 grid

    model = GraphWeatherForecaster(
        lat_lons,
        use_thermalizer=True,
        feature_dim=3,
        aux_dim=0,
        node_dim=256,
        num_blocks=1,
    )

    features = torch.randn(1, len(lat_lons), 3)
    pred = model(features, t=50)

    assert not torch.isnan(pred).any()
    assert torch.isfinite(pred).all()


def test_additional_thermalizer():
    """Basic sanity test for ThermalizerLayer with small input."""
    layer = ThermalizerLayer(input_dim=256)
    x = torch.randn(4, 256)
    t = torch.randint(0, 1000, (1,)).item()
    out = layer(x, t)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
