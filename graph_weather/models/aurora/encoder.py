"""
Swin 3D Transformer Encoder

- Uses a 3D convolution for initial feature extraction.
- Applies layer normalization and reshapes data.
- Uses a transformer-based encoder to learn spatial-temporal features.
"""

import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class Swin3DEncoder(nn.Module):
    """
    Encode 3D volumetric data into a sequence of per-voxel embeddings.

    A 3D convolution extracts local features, layer normalization is applied over the
    channel dimension, and the flattened voxel sequence is passed through the encoder
    stack of a ``torch.nn.Transformer``.
    """

    def __init__(self, in_channels=1, embed_dim=96):
        """
        Initialize the encoder.

        Args:
            in_channels (int): Number of channels in the input volume.
            embed_dim (int): Size of the embedding produced for each voxel.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.swin_transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=embed_dim * 4,
        )
        self.embed_dim = embed_dim

        # Define rearrangement patterns using einops
        self.to_transformer_format = Rearrange("b d h w c -> (d h w) b c")
        self.from_transformer_format = Rearrange("(d h w) b c -> b d h w c", d=None, h=None, w=None)

    # To use rearrange function directly instead of the Rearrange layer
    def forward(self, x):
        """
        Encode an input volume into a sequence of embeddings.

        Args:
            x (torch.Tensor): Input volume of shape (batch, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Encoded sequence of shape (batch, depth * height * width, embed_dim).
        """
        # 3D convolution with einops rearrangement
        x = self.conv1(x)

        # Rearrange for normalization using einops
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)

        # Store spatial dimensions for later reconstruction
        d, h, w = x.shape[1:4]

        # Transform to sequence format for transformer
        x = rearrange(x, "b d h w c -> (d h w) b c")
        x = self.swin_transformer.encoder(x)

        # Restore original spatial structure
        x = rearrange(x, "(d h w) b c -> b (d h w) c", d=d, h=h, w=w)

        # Reshape to the expected output format (batch, seq_len, embed_dim)
        x = rearrange(x, "b (d h w) c -> b (d h w) c", d=d, h=h, w=w)

        return x

    def convolution(self, x):
        """Apply 3D convolution with clear shape transformation."""
        return self.conv1(x)  # b c d h w -> b embed_dim d h w

    def normalization_layer(self, x):
        """Apply layer normalization with einops rearrangement."""
        x = rearrange(x, "b c d h w -> b d h w c")
        return self.norm(x)

    def transformer_encoder(self, x, spatial_dims):
        """
        Apply transformer encoding with proper shape handling.

        Args:
            x (torch.Tensor): Input tensor
            spatial_dims (tuple): Original (depth, height, width) dimensions
        """
        d, h, w = spatial_dims
        x = self.to_transformer_format(x)
        x = self.swin_transformer.encoder(x)
        x = self.from_transformer_format(x, d=d, h=h, w=w)
        return x
