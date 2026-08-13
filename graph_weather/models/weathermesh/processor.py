"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import torch.nn as nn
from natten import NeighborhoodAttention3D


@dataclass
class WeatherMeshProcessorConfig:
    """
    Configuration for `WeatherMeshProcessor`.

    Attributes:
        latent_dim: Channel width of the latent volume the processor operates on.
        n_layers: Number of stacked neighborhood attention layers.
        kernel: Neighborhood attention window as ``(depth, height, width)``.
        num_heads: Number of attention heads per layer.
    """

    latent_dim: int
    n_layers: int
    kernel: tuple
    num_heads: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshProcessor":
        """
        Build the processor configuration from a plain dictionary.

        Args:
            json: Mapping whose keys match the fields of this dataclass.

        Returns:
            The `WeatherMeshProcessorConfig` deserialized by dacite.
        """
        return dacite.from_dict(data_class=WeatherMeshProcessorConfig, data=json)

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


class WeatherMeshProcessor(nn.Module):
    """
    Processor that advances the latent state with 3D neighborhood attention.

    It is a plain stack of `NeighborhoodAttention3D` layers, each attending over a local
    ``kernel`` sized window in depth, height and width. Shape and channel width are unchanged,
    so one processor can be applied repeatedly to roll a forecast forward.
    """

    def __init__(self, latent_dim, n_layers=10, kernel=(5, 7, 7), num_heads=8):
        """
        Build the stack of neighborhood attention layers.

        Args:
            latent_dim: Channel width of the latent volume the processor operates on.
            n_layers: Number of stacked neighborhood attention layers.
            kernel: Neighborhood attention window as ``(depth, height, width)``.
            num_heads: Number of attention heads per layer.
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
        """
        Apply every neighborhood attention layer in sequence.

        Args:
            x: Latent tensor of shape ``(B, D, H, W, latent_dim)``.

        Returns:
            A tensor of the same shape as ``x``.
        """
        for layer in self.layers:
            x = layer(x)
        return x
