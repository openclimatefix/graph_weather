"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import einops
import torch
import torch.nn as nn
from natten import NeighborhoodAttention3D

from graph_weather.models.weathermesh.layers import ConvUpBlock


@dataclass
class WeatherMeshDecoderConfig:
    """
    Configuration for `WeatherMeshDecoder`.

    Attributes:
        latent_dim: Channel width of the latent volume fed to the decoder.
        output_channels_2d: Number of channels of the reconstructed surface field.
        output_channels_3d: Number of channels of the reconstructed pressure level field.
        n_conv_blocks: Number of `ConvUpBlock` stages in each of the two paths.
        hidden_dim: Base channel width. The latent is first widened to
            ``hidden_dim * 2 ** n_conv_blocks`` channels.
        kernel_size: Neighborhood attention window as ``(depth, height, width)``.
        num_heads: Number of attention heads per neighborhood attention layer.
        num_transformer_layers: Number of neighborhood attention layers applied to the latent.
    """

    latent_dim: int
    output_channels_2d: int
    output_channels_3d: int
    n_conv_blocks: int
    hidden_dim: int
    kernel_size: tuple
    num_heads: int
    num_transformer_layers: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshDecoder":
        """
        Build the decoder configuration from a plain dictionary.

        Args:
            json: Mapping whose keys match the fields of this dataclass.

        Returns:
            The `WeatherMeshDecoderConfig` deserialized by dacite.
        """
        return dacite.from_dict(data_class=WeatherMeshDecoderConfig, data=json)

    def to_json(self) -> dict:
        """
        Convert the configuration into a plain dictionary.

        Returns:
            A dictionary with one entry per dataclass field.
        """
        return dacite.asdict(self)


class WeatherMeshDecoder(nn.Module):
    """
    Decoder that turns a latent volume back into surface and pressure level fields.

    The latent is first refined by a stack of 3D neighborhood attention layers and widened by
    a 1x1x1 convolution. It is then split along the depth axis: every level but the last feeds
    the 3D upsampling path, and the last level is squeezed to 2D and feeds the surface path.
    Both paths are stacks of `ConvUpBlock` layers that reverse the encoder downsampling.
    """

    def __init__(
        self,
        latent_dim,
        output_channels_2d,
        output_channels_3d,
        n_conv_blocks=3,
        hidden_dim=256,
        kernel_size: tuple = (5, 7, 7),
        num_heads: int = 8,
        num_transformer_layers: int = 3,
    ):
        """
        Build the latent attention stack, the split projection and the two upsampling paths.

        Args:
            latent_dim: Channel width of the latent volume fed to the decoder.
            output_channels_2d: Number of channels of the reconstructed surface field.
            output_channels_3d: Number of channels of the reconstructed pressure level field.
            n_conv_blocks: Number of `ConvUpBlock` stages in each of the two paths.
            hidden_dim: Base channel width. The latent is widened to
                ``hidden_dim * 2 ** n_conv_blocks`` channels before the upsampling paths.
            kernel_size: Neighborhood attention window as ``(depth, height, width)``.
            num_heads: Number of attention heads per neighborhood attention layer.
            num_transformer_layers: Number of neighborhood attention layers applied to the
                latent.
        """
        super().__init__()

        # Transformer layers for initial decoding
        self.transformer_layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    embed_dim=latent_dim, num_heads=num_heads, kernel_size=kernel_size
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Split into pressure levels and surface paths
        self.split = nn.Conv3d(latent_dim, hidden_dim * (2**n_conv_blocks), kernel_size=1)

        # Pressure levels (3D) path
        self.pressure_path = nn.ModuleList(
            [
                ConvUpBlock(
                    hidden_dim * (2 ** (i + 1)),
                    hidden_dim * (2**i) if i > 0 else output_channels_3d,
                    is_3d=True,
                )
                for i in reversed(range(n_conv_blocks))
            ]
        )

        # Surface (2D) path
        self.surface_path = nn.ModuleList(
            [
                ConvUpBlock(
                    hidden_dim * (2 ** (i + 1)),
                    hidden_dim * (2**i) if i > 0 else output_channels_2d,
                )
                for i in reversed(range(n_conv_blocks))
            ]
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a latent volume into a surface field and a pressure level field.

        Args:
            latent: Latent tensor of shape ``(B, D, H, W, latent_dim)`` with depth ``D``
                covering the pressure levels plus one trailing surface level.

        Returns:
            A tuple ``(surface, pressure)``. ``surface`` has shape
            ``(B, output_channels_2d, H', W')`` and ``pressure`` has shape
            ``(B, output_channels_3d, D - 1, H', W')``, where ``H'`` and ``W'`` are the
            horizontal sizes after ``n_conv_blocks`` upsampling stages.
        """
        # Needs to be (B,D,H,W,C) with Batch, Depth (vertical levels), Height, Width, Channels
        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)

        latent = einops.rearrange(latent, "B D H W C -> B C D H W")
        # Split features
        features = self.split(latent)
        pressure_features = features[:, :, :-1]
        surface_features = features[:, :, -1:]
        # Decode pressure levels
        for block in self.pressure_path:
            pressure_features = block(pressure_features)
        # Decode surface features
        surface_features = surface_features.squeeze(2)
        for block in self.surface_path:
            surface_features = block(surface_features)

        return surface_features, pressure_features
