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

To make NATTEN work on a sphere, we implement our own circular padding. At the poles,
    we use the bump attention behavior from NATTEN. For position encoding of tokens,
    we use Rotary Embeddings.

In the default configuration of WeatherMesh 2, the NATTEN window is 5,7,7 in depth,
    width, height, corresponding to a physical size of 14 degrees longitude and
    latitude. WeatherMesh 2 contains two processors: a 6hr and a 1hr processor. Each
    is 10 NATTEN layers deep.

Training: distributed shampoo: https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md

Fork version of pytorch checkpoint library called matepoint to implement offloading to RAM

TODO: Add bump attention and rotary embeddings for the circular padding and position encoding

"""


@dataclass
class WeatherMeshConfig:
    """Configuration class for WeatherMesh model.

    Contains encoder, processors, decoder and other configuration parameters.
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
        """Create WeatherMeshConfig from JSON dictionary.

        Args:
            json: Dictionary containing configuration parameters

        Returns:
            WeatherMeshConfig instance
        """
        return dacite.from_dict(data_class=WeatherMeshConfig, data=json)

    def to_json(self) -> dict:
        """Convert WeatherMeshConfig to JSON dictionary.

        Returns:
            Dictionary containing configuration parameters
        """
        return dacite.asdict(self)


@dataclass
class WeatherMeshOutput:
    """Output container for WeatherMesh model predictions.

    Contains surface and pressure level predictions.
    """

    surface: torch.Tensor
    pressure: torch.Tensor


class WeatherMesh(nn.Module):
    """WeatherMesh model for weather forecasting using mesh-based processing.

    This model uses an encoder-processor-decoder architecture to process weather data
    on a mesh structure for improved forecasting accuracy.
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
        """Initialize the WeatherMesh model.

        Args:
            encoder: Encoder module or None to create default
            processors: List of processor modules or None to create default
            decoder: Decoder module or None to create default
            timesteps: List of timesteps for forecasting
            surface_channels: Number of surface channels
            pressure_channels: Number of pressure channels
            pressure_levels: Number of pressure levels
            latent_dim: Dimension of latent space
            encoder_num_conv_blocks: Number of convolution blocks in encoder
            encoder_num_transformer_layers: Number of transformer layers in encoder
            encoder_hidden_dim: Hidden dimension in encoder
            decoder_num_conv_blocks: Number of convolution blocks in decoder
            decoder_num_transformer_layers: Number of transformer layers in decoder
            decoder_hidden_dim: Hidden dimension in decoder
            processor_num_layers: Number of layers in processor
            kernel: Kernel size tuple
            num_heads: Number of attention heads
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
        """Forward pass of the WeatherMesh model.

        Args:
            surface: Surface weather data tensor
            pressure: Pressure level weather data tensor
            forecast_steps: Number of forecast steps to perform

        Returns:
            WeatherMeshOutput: Output containing surface and pressure predictions
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
