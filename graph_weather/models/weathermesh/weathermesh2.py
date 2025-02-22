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
        encoder: nn.Module,
        processors: List[nn.Module],
        decoder: nn.Module,
        timesteps: List[int],
    ):
        super().__init__()
        self.encoder = WeatherMeshEncoder(
            input_channels_2d=8, input_channels_3d=4, latent_dim=256, n_pressure_levels=25
        )
        self.processors = nn.ModuleList(WeatherMeshProcessor(latent_dim=256) for _ in timesteps)
        self.decoder = WeatherMeshDecoder(
            latent_dim=256, output_channels_2d=8, output_channels_3d=4, n_pressure_levels=25
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
