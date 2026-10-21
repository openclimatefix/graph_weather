"""Tests for the WeatherMesh model.

On CPU, NATTEN falls back to the flex-attention backend, which requires a head
dimension of at least 8 and materialises the full attention mask. These tests therefore
keep ``latent_dim // num_heads >= 8`` and use small grids so the mask stays small.
"""

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
        num_heads=1,
        hidden_dim=16,
        num_conv_blocks=3,
        num_transformer_layers=3,
    )
    x_2d = torch.randn(1, 2, 32, 64)
    x_3d = torch.randn(1, 1, 25, 32, 64)
    out = encoder(x_2d, x_3d)
    # The pressure path uses stride=(1, 2, 2) so the vertical depth is preserved: the
    # latent depth is the 25 pressure levels plus the single surface level.
    assert out.shape == (1, 26, 4, 8, 8)


def test_weathermesh_processor():
    processor = WeatherMeshProcessor(latent_dim=8, n_layers=2, num_heads=1)
    x = torch.randn(1, 6, 8, 16, 8)
    out = processor(x)
    assert out.shape == (1, 6, 8, 16, 8)


def test_weathermesh_decoder():
    decoder = WeatherMeshDecoder(
        latent_dim=16,
        output_channels_2d=8,
        output_channels_3d=4,
        kernel_size=(3, 3, 3),
        num_heads=2,
        hidden_dim=8,
        num_transformer_layers=1,
    )
    x = torch.randn(1, 6, 8, 16, 16)
    out = decoder(x)
    assert out[0].shape == (1, 8, 64, 128)
    assert out[1].shape == (1, 4, 5, 64, 128)


def test_weathermesh():
    model = WeatherMesh(
        encoder=None,
        processors=None,
        decoder=None,
        timesteps=[1, 6],
        surface_channels=8,
        pressure_channels=4,
        pressure_levels=5,
        latent_dim=8,
        encoder_num_conv_blocks=1,
        encoder_num_transformer_layers=1,
        encoder_hidden_dim=4,
        decoder_num_conv_blocks=1,
        decoder_num_transformer_layers=1,
        decoder_hidden_dim=4,
        processor_num_layers=2,
        kernel=(3, 5, 5),
        num_heads=1,
    )

    x_2d = torch.randn(1, 8, 32, 64)
    x_3d = torch.randn(1, 4, 5, 32, 64)
    out = model(x_2d, x_3d, forecast_steps=1)
    assert out.surface.shape == (1, 8, 32, 64)
    assert out.pressure.shape == (1, 4, 5, 32, 64)
