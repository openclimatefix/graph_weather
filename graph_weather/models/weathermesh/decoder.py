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
    """Configuration for WeatherMeshDecoder.
    
    Args:
        latent_dim: Dimension of the latent space
        output_channels_2d: Number of output channels for 2D surface data
        output_channels_3d: Number of output channels for 3D pressure level data
        n_conv_blocks: Number of convolutional blocks
        hidden_dim: Hidden dimension for the decoder
        kernel_size: Kernel size for the neighborhood attention
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer layers
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
    def from_json(json: dict) -> "WeatherMeshDecoderConfig":
        """Create a WeatherMeshDecoderConfig from a JSON dictionary.
        
        Args:
            json: Dictionary containing configuration values
        
        Returns:
            WeatherMeshDecoderConfig: Config object
        """
        return dacite.from_dict(data_class=WeatherMeshDecoderConfig, data=json)

    def to_json(self) -> dict:
        """Convert the config to a JSON dictionary.
        
        Returns:
            dict: Dictionary representation of the config
        """
        return dacite.asdict(self)


class WeatherMeshDecoder(nn.Module):
    """WeatherMesh decoder that transforms latent representations to 2D and 3D weather data.
    
    This decoder uses transformer layers followed by convolutional upsampling blocks
    to decode latent representations to surface (2D) and pressure level (3D) weather data.
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
        """Initialize the WeatherMeshDecoder.
        
        Args:
            latent_dim: Dimension of the latent space
            output_channels_2d: Number of output channels for 2D surface data
            output_channels_3d: Number of output channels for 3D pressure level data
            n_conv_blocks: Number of convolutional blocks
            hidden_dim: Hidden dimension for the decoder
            kernel_size: Kernel size for the neighborhood attention
            num_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
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
        """Decode latent representations to 2D and 3D weather data.
        
        Args:
            latent: Latent tensor of shape (B, D, H, W, C) where B is batch size,
                D is depth (vertical levels), H and W are height and width, and C is channels
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Surface features (2D) and pressure
                level features (3D)
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
