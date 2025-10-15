import torch
from torch import nn


class CaFAEncoder(nn.Module):
    """
    Encoder for CaFA
    This projects complex, high-resolution input weather state
    and transform it into a lower-resolution, high-dimensional
    latent representation that the processor can work with
    """

    def __init__(self, input_channels: int, model_dim: int, downsampling_factor: int = 1):
        """
        Args:
            input_channel: No. of channels/features in raw input data
            model_dim: Dimensions of the model's hidden layers (output channels)
            downsampling_factor: Factor to downsample the spatial dimensions by (i.e., 2 means H/2, W/2)
        """
        super().__init__()
        self.encoder = nn.Conv2d(
            in_channels=input_channels,
            out_channels=model_dim,
            kernel_size=downsampling_factor,
            stride=downsampling_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Encoded tensor of shape (batch, model_dim, height/downsampling_factor, width/downsampling_factor)
        """
        x = self.encoder(x)
        return x
