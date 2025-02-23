"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import torch.nn as nn
from natten import NeighborhoodAttention3D


@dataclass
class WeatherMeshProcessorConfig:
    latent_dim: int
    n_layers: int
    kernel: tuple
    num_heads: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshProcessor":
        return dacite.from_dict(data_class=WeatherMeshProcessorConfig, data=json)

    def to_json(self) -> dict:
        return dacite.asdict(self)


class WeatherMeshProcessor(nn.Module):
    def __init__(self, latent_dim, n_layers=10, kernel=(5, 7, 7), num_heads=8):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    dim=latent_dim,
                    num_heads=num_heads,
                    kernel_size=kernel,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
