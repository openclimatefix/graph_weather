"""
    Swin 3D Transformer Encoder:
    - Uses a 3D convolution for initial feature extraction.
    - Applies layer normalization and reshapes data.
    - Uses a transformer-based encoder to learn spatial-temporal features.
"""
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Swin3DEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96):
        """
        Initialize the Swin3DEncoder.
        Args:
            in_channels (int): Number of input channels (e.g., weather variable channels).
            embed_dim (int): Embedding dimension for the transformer.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.swin_transformer = nn.Transformer(embed_dim, num_encoder_layers=4)
    
    def forward(self, x):
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        Returns:
            torch.Tensor: Encoded features.
        """
        x = self.convolution(x)  # Convolution layer
        x = self.normalization_layer(x)  # Normalization layer
        x = self.transformer_encoder(x)  # Transformer encoder
        return x

    def convolution(self, x):
        """
        Applies 3D convolution to the input.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Convolved tensor.
        """
        return self.conv1(x)
    
    def normalization_layer(self, x):
        """
        Applies layer normalization after reshaping the data for transformer input.
        Args:
            x (torch.Tensor): Tensor after convolution.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Permute the tensor to move the channel dimension to the end
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        return self.norm(x)
    
    def transformer_encoder(self, x):
        """
        Applies the Swin Transformer encoder to the normalized data.
        Args:
            x (torch.Tensor): Normalized tensor.
        Returns:
            torch.Tensor: Transformed tensor with learned spatial-temporal features.
        """
        # Flatten the spatial dimensions for transformer processing
        x = x.view(x.shape[0], -1, x.shape[-1])  # (B, D*H*W, C)
        x = self.swin_transformer(x, x)  # Self-attention mechanism
        return x