import torch

from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder, WeatherMeshDecoderConfig
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder, WeatherMeshEncoderConfig
from graph_weather.models.weathermesh.processor import (
    WeatherMeshProcessor,
    WeatherMeshProcessorConfig,
)
from graph_weather.models.weathermesh.weathermesh2 import WeatherMesh, WeatherMeshConfig


def test_weathermesh_encoder():
    encoder = WeatherMeshEncoder(
        input_channels_2d=2,
        input_channels_3d=1,
        latent_dim=8,
        n_pressure_levels=25,
        kernel_size=(3, 3, 3),
        num_heads=2,
        hidden_dim=16,
        num_conv_blocks=3,
        num_transformer_layers=3,
    )
    x_2d = torch.randn(1, 2, 32, 64)
    x_3d = torch.randn(1, 1, 25, 32, 64)
    out = encoder(x_2d, x_3d)
    assert out.shape == (1, 5, 4, 8, 8)


def test_weathermesh_processor():
    processor = WeatherMeshProcessor(latent_dim=8, n_layers=2)
    x = torch.randn(1, 26, 32, 64, 8)
    out = processor(x)
    assert out.shape == (1, 26, 32, 64, 8)


def test_weathermesh_decoder():
    decoder = WeatherMeshDecoder(
        latent_dim=8,
        output_channels_2d=8,
        output_channels_3d=4,
        kernel_size=(3, 3, 3),
        num_heads=2,
        hidden_dim=8,
        num_transformer_layers=1,
    )
    x = torch.randn(1, 6, 32, 64, 8)
    out = decoder(x)
    assert out[0].shape == (1, 8, 256, 512)
    assert out[1].shape == (1, 4, 5, 256, 512)


def test_weathermesh():
    model = WeatherMesh(
        encoder=None,
        processors=None,
        decoder=None,
        timesteps=[1, 6],
        surface_channels=8,
        pressure_channels=4,
        pressure_levels=5,
        latent_dim=4,
        encoder_num_conv_blocks=1,
        encoder_num_transformer_layers=1,
        encoder_hidden_dim=4,
        decoder_num_conv_blocks=1,
        decoder_num_transformer_layers=1,
        decoder_hidden_dim=4,
        processor_num_layers=2,
        kernel=(3, 5, 5),
        num_heads=2,
    )

    x_2d = torch.randn(1, 8, 32, 64)
    x_3d = torch.randn(1, 4, 5, 32, 64)
    out = model(x_2d, x_3d, forecast_steps=1)
    assert out.surface.shape == (1, 8, 32, 64)
    assert out.pressure.shape == (1, 4, 5, 32, 64)
