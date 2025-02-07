"""
    3D Decoder:
    - Takes processed latent representations and reconstructs output.
    - Uses transposed convolution to upscale back to spatial-temporal format.
"""

import torch.nn as nn

class Decoder3D(nn.Module):
    """
    3D Decoder:
    - Takes processed latent representations and reconstructs the spatial-temporal output.
    - Uses transposed convolutions to upscale latent features to the original format.
    """
    def __init__(self, output_channels=1, embed_dim=96, target_shape=(32, 32, 32)):
        """
        Args:
            output_channels (int): Number of channels in the output tensor (e.g., 1 for grayscale).
            embed_dim (int): Dimension of the latent features (matches the encoder's output).
            target_shape (tuple): The desired shape of the reconstructed 3D tensor (D, H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.target_shape = target_shape
        self.deconv1 = nn.ConvTranspose3d(
            embed_dim, output_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x):
        """
        Forward pass for the decoder.
        Args:
            x (torch.Tensor): Input latent representation, shape (batch, seq_len, embed_dim).
        Returns:
            torch.Tensor: Reconstructed 3D tensor, shape (batch, output_channels, *target_shape).
        """
        batch_size = x.shape[0]
        depth, height, width = self.target_shape
        # Reshape latent features into 3D tensor
        x = x.view(batch_size, self.embed_dim, depth, height, width)
        # Transposed convolution to upscale to the final shape
        x = self.deconv1(x)
        return x