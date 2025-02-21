"""
Swin 3D Transformer Encoder:
- Uses a 3D convolution for initial feature extraction.
- Applies layer normalization and reshapes data.
- Uses a transformer-based encoder to learn spatial-temporal features.
"""

import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class Swin3DEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.swin_transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=8,  # Standard number of heads
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=embed_dim * 4
        )
        self.embed_dim = embed_dim
    def forward(self, x):
        # Apply 3D convolution: (batch, channels, depth, height, width) -> (batch, embed_dim, depth, height, width)
        x = self.conv1(x)
        
        # Reshape for layer norm: (batch, embed_dim, depth, height, width) -> (batch, depth*height*width, embed_dim)
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # -> (batch, depth, height, width, embed_dim)
        x = x.view(batch_size, -1, self.embed_dim)  # -> (batch, depth*height*width, embed_dim)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Apply transformer
        # For transformer, input shape should be (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.swin_transformer.encoder(x)
        x = x.transpose(0, 1)  # Back to (batch, seq_len, embed_dim)
        
        return x

    def convolution(self, x):
        """
        Applies 3D convolution to the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Convolved tensor.
        """
        # b c d h w -> b embed_dim d h w
        return self.conv1(x)

    def normalization_layer(self, x):
        """
        Applies layer normalization after reshaping the data for transformer input.

        Args:
            x (torch.Tensor): Tensor after convolution.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Rearrange for normalization: b c d h w -> b d h w c
        x = rearrange(x, 'b c d h w -> b d h w c')
        # Apply layer normalization on the last dimension (embed_dim)
        x = self.norm(x)
        return x

    def transformer_encoder(self, x):
        """
        Applies the Swin Transformer encoder to the normalized data.

        Args:
            x (torch.Tensor): Normalized tensor.

        Returns:
            torch.Tensor: Transformed tensor with learned spatial-temporal features.
        """
        # Flatten spatial dimensions: b d h w c -> b (d h w) c
        x = self.to_transformer_format(x)
        
        # Apply transformer
        x = self.swin_transformer(x, x)
        
        return x