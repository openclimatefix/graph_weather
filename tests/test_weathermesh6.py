import torch

from graph_weather.models.weathermesh.weathermesh6 import (
    WeatherMesh6,
    WeatherMesh6Config,
    WeatherMesh6Output,
)


def test_weathermesh6_forward():
    model = WeatherMesh6(
        resolution=0.25,
        surface_channels=4,
        pressure_channels=2,
        pressure_levels=5,
        latent_dim=8,
        hidden_dim=8,
        kernel=(3, 3, 3),
        num_heads=1,
        encoder_num_conv_blocks=1,
        encoder_num_transformer_layers=1,
        encoder_hidden_dim=8,
        decoder_num_conv_blocks=1,
        decoder_num_transformer_layers=1,
        decoder_hidden_dim=8,
        processor_num_layers=2,
        timesteps=[1, 6],
        ensemble_size=1,
        use_matepoint=False,
    )

    x_2d = torch.randn(1, 4, 32, 64)
    x_3d = torch.randn(1, 2, 5, 32, 64)
    out = model(x_2d, x_3d, forecast_steps=1)
    assert isinstance(out, WeatherMesh6Output)
    assert out.surface.shape == (1, 4, 32, 64)
    assert out.pressure.shape == (1, 2, 5, 32, 64)


def test_weathermesh6_latent_rollout():
    model = WeatherMesh6(
        resolution=0.25,
        surface_channels=4,
        pressure_channels=2,
        pressure_levels=5,
        latent_dim=8,
        hidden_dim=8,
        kernel=(3, 3, 3),
        num_heads=1,
        encoder_num_conv_blocks=1,
        encoder_num_transformer_layers=1,
        encoder_hidden_dim=8,
        decoder_num_conv_blocks=1,
        decoder_num_transformer_layers=1,
        decoder_hidden_dim=8,
        processor_num_layers=2,
        timesteps=[1, 6],
        ensemble_size=1,
        use_matepoint=False,
    )

    x_2d = torch.randn(1, 4, 16, 32)
    x_3d = torch.randn(1, 2, 5, 16, 32)
    out = model(x_2d, x_3d, forecast_steps=3)
    assert out.surface.shape == (1, 4, 16, 32)
    assert out.pressure.shape == (1, 2, 5, 16, 32)


def test_weathermesh6_ensemble():
    model = WeatherMesh6(
        resolution=0.25,
        surface_channels=4,
        pressure_channels=2,
        pressure_levels=5,
        latent_dim=8,
        hidden_dim=8,
        kernel=(3, 3, 3),
        num_heads=1,
        encoder_num_conv_blocks=1,
        encoder_num_transformer_layers=1,
        encoder_hidden_dim=8,
        decoder_num_conv_blocks=1,
        decoder_num_transformer_layers=1,
        decoder_hidden_dim=8,
        processor_num_layers=2,
        timesteps=[1, 6],
        ensemble_size=4,
        use_matepoint=False,
    )

    x_2d = torch.randn(1, 4, 16, 32)
    x_3d = torch.randn(1, 2, 5, 16, 32)
    out = model(x_2d, x_3d, forecast_steps=1, ensemble_size=4)
    assert out.surface.shape == (1, 4, 4, 16, 32)
    assert out.pressure.shape == (1, 4, 2, 5, 16, 32)


def test_weathermesh6_config_roundtrip():
    config = WeatherMesh6Config(
        resolution=0.1,
        surface_channels=8,
        pressure_channels=4,
        pressure_levels=25,
        ensemble_size=1,
    )
    json_data = config.to_json()
    restored = WeatherMesh6Config.from_json(json_data)
    assert restored.resolution == 0.1
    assert restored.ensemble_size == 1
    assert restored.timesteps == [1, 6]
