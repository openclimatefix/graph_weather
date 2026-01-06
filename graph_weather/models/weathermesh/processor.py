"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import torch.nn as nn
from natten import NeighborhoodAttention3D


@dataclass
class WeatherMeshProcessorConfig:
    """Configuration class for WeatherMesh processor.

    Contains parameters for configuring the WeatherMesh processor.
    """

    latent_dim: int
    n_layers: int
    kernel: tuple
    num_heads: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshProcessor":
        """Create WeatherMeshProcessorConfig from JSON dictionary.

        Args:
            json: Dictionary containing configuration parameters

        Returns:
            WeatherMeshProcessorConfig instance
        """
        return dacite.from_dict(data_class=WeatherMeshProcessorConfig, data=json)

    def to_json(self) -> dict:
        """Convert WeatherMeshProcessorConfig to JSON dictionary.

        Returns:
            Dictionary containing configuration parameters
        """
        return dacite.asdict(self)


class WeatherMeshProcessor(nn.Module):
    """WeatherMesh processor module using neighborhood attention.

    Processes latent representations using multiple neighborhood attention layers.
    """

    def __init__(self, latent_dim, n_layers=10, kernel=(5, 7, 7), num_heads=8):
        """Initialize the WeatherMesh processor.

        Args:
            latent_dim: Dimension of the latent space
            n_layers: Number of processor layers
            kernel: Kernel size for neighborhood attention
            num_heads: Number of attention heads
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    kernel_size=kernel,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        """Forward pass of the WeatherMesh processor.

        Args:
            x: Input tensor to process

        Returns:
            Processed tensor after applying all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x
