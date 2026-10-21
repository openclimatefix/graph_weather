"""Tests for the WeatherMesh model.

On CPU, NATTEN falls back to the flex-attention backend, which requires a head
dimension of at least 8 and materialises the full attention mask. These tests therefore
keep ``latent_dim // num_heads >= 8`` and use small grids so the mask stays small.
"""

import json

import torch
import torch.nn as nn

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


class _TinyProcessor(nn.Module):
    """Stand-in processor that owns parameters, so registration is observable."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _weathermesh_with(processors):
    """Build a WeatherMesh from pre-built parts, bypassing the NATTEN-backed defaults."""
    return WeatherMesh(
        encoder=nn.Identity(),
        processors=processors,
        decoder=nn.Identity(),
        timesteps=[1, 6],
        surface_channels=None,
        pressure_channels=None,
        pressure_levels=None,
        latent_dim=None,
        encoder_num_conv_blocks=None,
        encoder_num_transformer_layers=None,
        encoder_hidden_dim=None,
        decoder_num_conv_blocks=None,
        decoder_num_transformer_layers=None,
        decoder_hidden_dim=None,
        processor_num_layers=None,
        kernel=None,
        num_heads=None,
    )


def test_weathermesh_supplied_processors_are_registered():
    processors = [_TinyProcessor(), _TinyProcessor()]
    expected_params = sum(p.numel() for m in processors for p in m.parameters())

    model = _weathermesh_with(processors)

    assert isinstance(model.processors, nn.ModuleList)
    assert len(model.processors) == 2

    registered = sum(
        p.numel() for name, p in model.named_parameters() if name.startswith("processors.")
    )
    assert registered == expected_params

    assert [name for name in model.state_dict() if name.startswith("processors.")] == [
        "processors.0.linear.weight",
        "processors.0.linear.bias",
        "processors.1.linear.weight",
        "processors.1.linear.bias",
    ]

    model.eval()
    assert all(not processor.training for processor in processors)

    model.to(torch.float64)
    assert all(processor.linear.weight.dtype == torch.float64 for processor in processors)


def test_weathermesh_default_processors_are_registered():
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

    assert isinstance(model.processors, nn.ModuleList)
    assert len(model.processors) == 2
    assert any(name.startswith("processors.") for name in model.state_dict())


def test_weathermesh_configs_round_trip():
    encoder_config = WeatherMeshEncoderConfig(
        input_channels_2d=8,
        input_channels_3d=4,
        latent_dim=8,
        n_pressure_levels=5,
        num_conv_blocks=1,
        hidden_dim=4,
        kernel_size=(3, 5, 5),
        num_heads=1,
        num_transformer_layers=1,
    )
    decoder_config = WeatherMeshDecoderConfig(
        latent_dim=8,
        output_channels_2d=8,
        output_channels_3d=4,
        n_conv_blocks=1,
        hidden_dim=4,
        kernel_size=(3, 5, 5),
        num_heads=1,
        num_transformer_layers=1,
    )
    processor_config = WeatherMeshProcessorConfig(
        latent_dim=8, n_layers=2, kernel=(3, 5, 5), num_heads=1
    )
    config = WeatherMeshConfig(
        encoder=encoder_config,
        processors=[processor_config],
        decoder=decoder_config,
        timesteps=[1],
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

    for original, config_class in (
        (encoder_config, WeatherMeshEncoderConfig),
        (decoder_config, WeatherMeshDecoderConfig),
        (processor_config, WeatherMeshProcessorConfig),
        (config, WeatherMeshConfig),
    ):
        as_json = original.to_json()
        assert isinstance(as_json, dict)
        assert config_class.from_json(as_json) == original

        # Through real JSON as well, where the tuple fields come back as lists.
        through_json = json.loads(json.dumps(as_json))
        assert config_class.from_json(through_json) == original

    nested = config.to_json()
    assert isinstance(nested["encoder"], dict)
    assert isinstance(nested["processors"][0], dict)
