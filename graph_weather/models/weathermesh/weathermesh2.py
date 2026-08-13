"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass
from typing import List

import dacite
import torch
import torch.nn as nn

from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder, WeatherMeshDecoderConfig
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder, WeatherMeshEncoderConfig
from graph_weather.models.weathermesh.processor import (
    WeatherMeshProcessor,
    WeatherMeshProcessorConfig,
)

"""
Notes on implementation

To make NATTEN work on a sphere, we implement our own circular padding. At the poles, we use
the bump attention behavior from NATTEN. For position encoding of tokens, we use Rotary
Embeddings.

In the default configuration of WeatherMesh 2, the NATTEN window is 5,7,7 in depth, height,
and width, corresponding to a physical size of 14 degrees longitude and latitude.
WeatherMesh 2
contains two processors: a 6hr and a 1hr processor. Each is 10 NATTEN layers deep.

Training: distributed shampoo: https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md

Fork version of pytorch checkpoint library called matepoint to implement offloading to RAM

TODO: Add bump attention and rotary embeddings for the circular padding and position encoding

"""


@dataclass
class WeatherMeshConfig:
    """
    Configuration for the full `WeatherMesh` model.

    Holds the configs of the three sub-modules plus the flat set of values used to build
    default sub-modules when none are supplied.

    Attributes:
        encoder: Configuration of the encoder.
        processors: One configuration per processor.
        decoder: Configuration of the decoder.
        timesteps: Lead times, in hours, covered by the processors; one entry per processor.
        surface_channels: Number of channels of the surface (2D) fields.
        pressure_channels: Number of channels of the pressure level (3D) fields.
        pressure_levels: Number of pressure levels of the 3D fields.
        latent_dim: Channel width of the latent space shared by all three sub-modules.
        encoder_num_conv_blocks: Number of downsampling stages per encoder path.
        encoder_num_transformer_layers: Number of encoder neighborhood attention layers.
        encoder_hidden_dim: Base channel width of the encoder convolution stages.
        decoder_num_conv_blocks: Number of upsampling stages per decoder path.
        decoder_num_transformer_layers: Number of decoder neighborhood attention layers.
        decoder_hidden_dim: Base channel width of the decoder convolution stages.
        processor_num_layers: Number of neighborhood attention layers per processor.
        kernel: Neighborhood attention window as ``(depth, height, width)``.
        num_heads: Number of attention heads per neighborhood attention layer.
    """

    encoder: WeatherMeshEncoderConfig
    processors: List[WeatherMeshProcessorConfig]
    decoder: WeatherMeshDecoderConfig
    timesteps: List[int]
    surface_channels: int
    pressure_channels: int
    pressure_levels: int
    latent_dim: int
    encoder_num_conv_blocks: int
    encoder_num_transformer_layers: int
    encoder_hidden_dim: int
    decoder_num_conv_blocks: int
    decoder_num_transformer_layers: int
    decoder_hidden_dim: int
    processor_num_layers: int
    kernel: tuple
    num_heads: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMesh":
        """
        Build the model configuration from a plain dictionary.

        Args:
            json: Mapping whose keys match the fields of this dataclass.

        Returns:
            The `WeatherMeshConfig` deserialized by dacite.
        """
        return dacite.from_dict(data_class=WeatherMeshConfig, data=json)

    def to_json(self) -> dict:
        """
        Convert the configuration into a plain dictionary.

        Returns:
            A dictionary with one entry per dataclass field.

        Note:
            ``dacite`` does not provide ``asdict``, so this currently raises
            ``AttributeError``. Flagged in the pull request that added this
            docstring; the change itself touches docstrings only.
        """
        return dacite.asdict(self)


@dataclass
class WeatherMeshOutput:
    """
    Result of a `WeatherMesh` forward pass.

    Attributes:
        surface: Surface field of shape ``(B, surface_channels, H, W)``.
        pressure: Pressure level field of shape ``(B, pressure_channels, D, H, W)``.
    """

    surface: torch.Tensor
    pressure: torch.Tensor


class WeatherMesh(nn.Module):
    """
    WeatherMesh forecasting model: encoder, one or more processors and a decoder.

    The encoder maps the surface and pressure level fields into a single latent volume, the
    processors advance that latent with 3D neighborhood attention, and the decoder maps it
    back to surface and pressure level fields. Each sub-module can be passed in ready-made;
    otherwise it is built from the remaining arguments.
    """

    def __init__(
        self,
        encoder: nn.Module | None,
        processors: List[nn.Module] | None,
        decoder: nn.Module | None,
        timesteps: List[int],
        surface_channels: int | None,
        pressure_channels: int | None,
        pressure_levels: int | None,
        latent_dim: int | None,
        encoder_num_conv_blocks: int | None,
        encoder_num_transformer_layers: int | None,
        encoder_hidden_dim: int | None,
        decoder_num_conv_blocks: int | None,
        decoder_num_transformer_layers: int | None,
        decoder_hidden_dim: int | None,
        processor_num_layers: int | None,
        kernel: tuple | None,
        num_heads: int | None,
    ):
        """
        Assemble the model, building any sub-module that was not supplied.

        Args:
            encoder: Ready-made encoder. If None, a `WeatherMeshEncoder` is built from the
                ``surface_channels``, ``pressure_channels``, ``pressure_levels``,
                ``latent_dim``, ``encoder_*``, ``kernel`` and ``num_heads`` arguments.
            processors: Ready-made processors, one per entry of ``timesteps``. If None, one
                `WeatherMeshProcessor` per timestep is built from ``latent_dim``,
                ``processor_num_layers``, ``kernel`` and ``num_heads``. Either way the
                processors are kept in a plain list rather than an `nn.ModuleList`.
            decoder: Ready-made decoder. If None, a `WeatherMeshDecoder` is built from the
                ``latent_dim``, ``surface_channels``, ``pressure_channels``, ``decoder_*``,
                ``kernel`` and ``num_heads`` arguments.
            timesteps: Lead times, in hours, covered by the processors; its length sets how
                many processors are built and is checked against ``processors``.
            surface_channels: Number of channels of the surface (2D) fields.
            pressure_channels: Number of channels of the pressure level (3D) fields.
            pressure_levels: Number of pressure levels of the 3D fields.
            latent_dim: Channel width of the latent space shared by all three sub-modules.
            encoder_num_conv_blocks: Number of downsampling stages per encoder path.
            encoder_num_transformer_layers: Number of encoder neighborhood attention layers.
            encoder_hidden_dim: Base channel width of the encoder convolution stages.
            decoder_num_conv_blocks: Number of upsampling stages per decoder path.
            decoder_num_transformer_layers: Number of decoder neighborhood attention layers.
            decoder_hidden_dim: Base channel width of the decoder convolution stages.
            processor_num_layers: Number of neighborhood attention layers per processor.
            kernel: Neighborhood attention window as ``(depth, height, width)``.
            num_heads: Number of attention heads per neighborhood attention layer.

        Raises:
            AssertionError: If ``processors`` is supplied and its length differs from the
                length of ``timesteps``.
        """
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = WeatherMeshEncoder(
                input_channels_2d=surface_channels,
                input_channels_3d=pressure_channels,
                latent_dim=latent_dim,
                n_pressure_levels=pressure_levels,
                num_conv_blocks=encoder_num_conv_blocks,
                hidden_dim=encoder_hidden_dim,
                kernel_size=kernel,
                num_heads=num_heads,
                num_transformer_layers=encoder_num_transformer_layers,
            )
        if processors is not None:
            assert len(processors) == len(
                timesteps
            ), "Number of processors must match number of timesteps"
            self.processors = processors
        else:
            self.processors = [
                WeatherMeshProcessor(
                    latent_dim=latent_dim,
                    n_layers=processor_num_layers,
                    kernel=kernel,
                    num_heads=num_heads,
                )
                for _ in range(len(timesteps))
            ]
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = WeatherMeshDecoder(
                latent_dim=latent_dim,
                output_channels_2d=surface_channels,
                output_channels_3d=pressure_channels,
                n_conv_blocks=decoder_num_conv_blocks,
                hidden_dim=decoder_hidden_dim,
                kernel_size=kernel,
                num_heads=num_heads,
                num_transformer_layers=decoder_num_transformer_layers,
            )
        self.timesteps = timesteps

    def forward(
        self, surface: torch.Tensor, pressure: torch.Tensor, forecast_steps: int
    ) -> WeatherMeshOutput:
        """
        Roll the forecast forward and decode the final latent state.

        The inputs are encoded once. Every processor is then applied in sequence, and that
        whole sequence is repeated ``forecast_steps`` times, so the lead time advanced is
        ``forecast_steps`` times the sum of ``self.timesteps``. Only the final state is
        decoded; intermediate states are not returned.

        Args:
            surface: Surface tensor of shape ``(B, surface_channels, H, W)``.
            pressure: Pressure level tensor of shape ``(B, pressure_channels, D, H, W)``.
            forecast_steps: Number of times the full processor sequence is applied.

        Returns:
            A `WeatherMeshOutput` holding the decoded surface and pressure level fields.
        """
        # Encode input
        latent = self.encoder(surface, pressure)

        # Apply processors for each forecast step
        for _ in range(forecast_steps):
            for processor in self.processors:
                latent = processor(latent)

        # Decode output
        surface_out, pressure_out = self.decoder(latent)

        return WeatherMeshOutput(surface=surface_out, pressure=pressure_out)
