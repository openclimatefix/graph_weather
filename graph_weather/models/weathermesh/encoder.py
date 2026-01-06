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
    """Configuration for WeatherMeshEncoder.
    
    Args:
        input_channels_2d: Number of input channels for 2D surface data
        input_channels_3d: Number of input channels for 3D pressure level data
        latent_dim: Dimension of the latent space
        n_pressure_levels: Number of pressure levels
        num_conv_blocks: Number of convolutional blocks
        hidden_dim: Hidden dimension for the encoder
        kernel_size: Kernel size for the neighborhood attention
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer layers
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
    def from_json(json: dict) -> "WeatherMeshEncoderConfig":
        """Create a WeatherMeshEncoderConfig from a JSON dictionary.
        
        Args:
            json: Dictionary containing configuration values
        
        Returns:
            WeatherMeshEncoderConfig: Config object
        """
        return dacite.from_dict(data_class=WeatherMeshEncoderConfig, data=json)

    def to_json(self) -> dict:
        """Convert the config to a JSON dictionary.
        
        Returns:
            dict: Dictionary representation of the config
        """
        return dacite.asdict(self)


class WeatherMeshEncoder(nn.Module):
    """WeatherMesh encoder that transforms 2D and 3D weather data to latent representations.
    
    This encoder uses convolutional downsampling blocks followed by transformer layers
    to encode surface (2D) and pressure level (3D) weather data to latent representations.
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
        """Initialize the WeatherMeshEncoder.
        
        Args:
            input_channels_2d: Number of input channels for 2D surface data
            input_channels_3d: Number of input channels for 3D pressure level data
            latent_dim: Dimension of the latent space
            n_pressure_levels: Number of pressure levels
            num_conv_blocks: Number of convolutional blocks
            hidden_dim: Hidden dimension for the encoder
            kernel_size: Kernel size for the neighborhood attention
            num_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
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
        """Encode 2D and 3D weather data to latent representations.
        
        Args:
            surface: Surface tensor of shape (B, C_2d, H, W)
            pressure: Pressure level tensor of shape (B, C_3d, D, H, W)
        
        Returns:
            torch.Tensor: Latent tensor of shape (B, D, H, W, C_latent)
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
