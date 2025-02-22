from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder
from graph_weather.models.weathermesh.processor import WeatherMeshProcessor
from graph_weather.models.weathermesh.weathermesh2 import WeatherMesh
import torch
import numpy as np


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
    processor = WeatherMeshProcessor(latent_dim=256)
    x = torch.randn(1, 256, 25, 32, 64)
    out = processor(x)
    assert out.shape == (1, 256, 25, 32, 64)


def test_weathermesh_decoder():
    decoder = WeatherMeshDecoder(
        latent_dim=256, output_channels_2d=8, output_channels_3d=4, n_pressure_levels=25
    )
    x = torch.randn(1, 256, 25, 32, 64)
    out = decoder(x)
    assert out[0].shape == (1, 8, 32, 64)
    assert out[1].shape == (1, 4, 25, 32, 64)


def test_weathermesh():
    encoder = WeatherMeshEncoder(
        input_channels_2d=8, input_channels_3d=4, latent_dim=256, n_pressure_levels=25
    )
    processors = [WeatherMeshProcessor(latent_dim=256) for _ in range(10)]
    decoder = WeatherMeshDecoder(
        latent_dim=256, output_channels_2d=8, output_channels_3d=4, n_pressure_levels=25
    )
    model = WeatherMesh(encoder, processors, decoder, timesteps=[1, 6])
    x_2d = torch.randn(1, 8, 32, 64)
    x_3d = torch.randn(1, 4, 25, 32, 64)
    out = model(x_2d, x_3d, forecast_steps=1)
    assert out[0].shape == (1, 8, 32, 64)
    assert out[1].shape == (1, 4, 25, 32, 64)
