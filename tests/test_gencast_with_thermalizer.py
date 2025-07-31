import torch
from graph_weather.models.forecast import GraphWeatherForecaster
from graph_weather.models.layers.thermalizer import ThermalizerLayer


def test_gencast_thermal_integration():
    # End-to-end test: GraphWeatherForecaster with thermalizer
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

    # Output shape should match (grid or node layout)
    assert not torch.isnan(pred).any()
    assert torch.isfinite(pred).all()

    if pred.dim() == 4:
        assert pred.shape[0] == features.shape[0]
        assert pred.shape[2] * pred.shape[3] == features.shape[1]
    else:
        assert pred.shape == features.shape


def test_thermalizer_small_grids():
    # Test thermalizer layer with different small grids
    layer = ThermalizerLayer(input_dim=256)
    t = torch.randint(0, 1000, (1,)).item()

    for nodes in [4, 9, 64]:  # 2x2, 3x3, 8x8
        x = torch.randn(nodes, 256)
        out = layer(x, t)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


def test_small_grid_integration():
    # 2x2 integration test (GraphWeatherForecaster + Thermalizer)
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
    # Simple sanity test for thermalizer alone
    layer = ThermalizerLayer(input_dim=256)
    x = torch.randn(4, 256)
    t = torch.randint(0, 1000, (1,)).item()
    out = layer(x, t)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
