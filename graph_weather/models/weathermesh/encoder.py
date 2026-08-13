"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import einops
import torch
import torch.nn as nn
from natten import NeighborhoodAttention3D

from graph_weather.models.weathermesh.layers import ConvDownBlock


@dataclass
class WeatherMeshEncoderConfig:
    """
    Configuration for `WeatherMeshEncoder`.

    Attributes:
        input_channels_2d: Number of channels of the surface (2D) input.
        input_channels_3d: Number of channels of the pressure level (3D) input.
        latent_dim: Channel width of the latent space the encoder projects into.
        n_pressure_levels: Number of pressure levels of the 3D input. Kept as part of the
            configuration; the encoder currently accepts it without using it to size any
            layer, because the 3D path preserves depth.
        num_conv_blocks: Number of `ConvDownBlock` stages in each of the two paths.
        hidden_dim: Base channel width. Stage ``i`` outputs ``hidden_dim * 2 ** (i + 1)``
            channels.
        kernel_size: Neighborhood attention window as ``(depth, height, width)``.
        num_heads: Number of attention heads per neighborhood attention layer.
        num_transformer_layers: Number of neighborhood attention layers applied to the latent.
    """

    input_channels_2d: int
    input_channels_3d: int
    latent_dim: int
    n_pressure_levels: int
    num_conv_blocks: int
    hidden_dim: int
    kernel_size: tuple
    num_heads: int
    num_transformer_layers: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshEncoder":
        """
        Build the encoder configuration from a plain dictionary.

        Args:
            json: Mapping whose keys match the fields of this dataclass.

        Returns:
            The `WeatherMeshEncoderConfig` deserialized by dacite.
        """
        return dacite.from_dict(data_class=WeatherMeshEncoderConfig, data=json)

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


class WeatherMeshEncoder(nn.Module):
    """
    Encoder that maps surface and pressure level fields into a single latent volume.

    Surface data is downsampled by a stack of 2D `ConvDownBlock` layers and pressure level
    data by an equivalent stack of 3D blocks that use ``stride=(1, 2, 2)`` so that the
    vertical dimension is preserved. The surface features are appended to the pressure
    features as one extra level along the depth axis, projected to ``latent_dim`` by a 1x1x1
    convolution and refined by a stack of 3D neighborhood attention layers.
    """

    def __init__(
        self,
        input_channels_2d: int,
        input_channels_3d: int,
        latent_dim: int,
        n_pressure_levels: int,
        num_conv_blocks: int = 3,
        hidden_dim: int = 256,
        kernel_size: tuple = (5, 7, 7),
        num_heads: int = 8,
        num_transformer_layers: int = 3,
    ):
        """
        Build the surface path, the pressure level path and the latent attention stack.

        Args:
            input_channels_2d: Number of channels of the surface (2D) input.
            input_channels_3d: Number of channels of the pressure level (3D) input.
            latent_dim: Channel width of the latent space the encoder projects into.
            n_pressure_levels: Number of pressure levels of the 3D input. Accepted as part of
                the interface; it is not used to size any layer, because the 3D path keeps the
                depth of its input.
            num_conv_blocks: Number of `ConvDownBlock` stages in each of the two paths.
            hidden_dim: Base channel width. Stage ``i`` outputs ``hidden_dim * 2 ** (i + 1)``
                channels.
            kernel_size: Neighborhood attention window as ``(depth, height, width)``.
            num_heads: Number of attention heads per neighborhood attention layer.
            num_transformer_layers: Number of neighborhood attention layers applied to the
                latent.
        """
        super().__init__()

        # Surface (2D) path
        self.surface_path = nn.ModuleList(
            [
                ConvDownBlock(
                    input_channels_2d if i == 0 else hidden_dim * (2**i),
                    hidden_dim * (2 ** (i + 1)),
                )
                for i in range(num_conv_blocks)
            ]
        )

        # Pressure levels (3D) path
        self.pressure_path = nn.ModuleList(
            [
                ConvDownBlock(
                    input_channels_3d if i == 0 else hidden_dim * (2**i),
                    hidden_dim * (2 ** (i + 1)),
                    stride=(1, 2, 2),  # Want to keep depth the same size
                    is_3d=True,
                )
                for i in range(num_conv_blocks)
            ]
        )

        # Transformer layers for final encoding
        self.transformer_layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    embed_dim=latent_dim, kernel_size=kernel_size, num_heads=num_heads
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Final projection to latent space
        self.to_latent = nn.Conv3d(hidden_dim * (2**num_conv_blocks), latent_dim, kernel_size=1)

    def forward(self, surface: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        """
        Encode one surface field and one pressure level field into a latent volume.

        Args:
            surface: Surface tensor of shape ``(B, input_channels_2d, H, W)``.
            pressure: Pressure level tensor of shape ``(B, input_channels_3d, D, H, W)``.

        Returns:
            Latent tensor of shape ``(B, D + 1, H', W', latent_dim)``, where ``H'`` and ``W'``
            are the horizontal sizes after ``num_conv_blocks`` downsampling stages and the
            extra level along the depth axis carries the encoded surface features.
        """
        # Process surface data
        for block in self.surface_path:
            surface = block(surface)

        # Process pressure level data
        for block in self.pressure_path:
            pressure = block(pressure)
        # Combine features
        features = torch.cat(
            [pressure, surface.unsqueeze(2)], dim=2
        )  # B C D H W currently, want it to be B D H W C

        # Transform to latent space
        latent = self.to_latent(features)

        # Reshape to get the shapes
        latent = einops.rearrange(latent, "B C D H W -> B D H W C")
        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)
        return latent
