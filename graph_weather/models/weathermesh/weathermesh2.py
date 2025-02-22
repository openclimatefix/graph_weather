"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder
from graph_weather.models.weathermesh.processor import WeatherMeshProcessor

"""
Notes on implementation

To make NATTEN work on a sphere, we implement our own circular padding. At the poles, we use the bump attention behavior from NATTEN. For position encoding of tokens, we use Rotary Embeddings.

In the default configuration of WeatherMesh 2, the NATTEN window is 5,7,7 in depth, width, height, corresponding to a physical size of 14 degrees longitude and latitude. WeatherMesh 2 contains two processors: a 6hr and a 1hr processor. Each is 10 NATTEN layers deep.

Training: distributed shampoo: https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md

Fork version of pytorch checkpoint library called matepoint to implement offloading to RAM

"""


class WeatherMesh(nn.Module):
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
        super().__init__()
        assert len(processors) == len(timesteps)
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
            self.processors = processors
        else:
            self.processors = [
                WeatherMeshProcessor(latent_dim=latent_dim, n_layers=processor_num_layers, kernel=kernel, num_heads=num_heads)
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
        self, x_2d: torch.Tensor, x_3d: torch.Tensor, forecast_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode input
        latent = self.encoder(x_2d, x_3d)

        # Apply processors for each forecast step
        for _ in range(forecast_steps):
            for processor in self.processors:
                latent = processor(latent)

        # Decode output
        surface_out, pressure_out = self.decoder(latent)

        return surface_out, pressure_out
