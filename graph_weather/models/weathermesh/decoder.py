"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""
import torch
import torch.nn as nn
from natten import NeighborhoodAttention3D

from graph_weather.models.weathermesh.layers import ConvUpBlock


class WeatherMeshDecoder(nn.Module):
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
        super().__init__()

        # Transformer layers for initial decoding
        self.transformer_layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    dim=latent_dim, num_heads=num_heads, kernel_size=kernel_size
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
        # Needs to be (B,D,H,W,C) with Batch, Depth (vertical levels), Height, Width, Channels

        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)

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
