import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    A generic decoder that can apply upsampling or final transformations
    to match the desired output dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., predicted weather variables).
        upsample_factor (int): Factor to upsample spatial dimensions.
        num_layers (int): Number of convolutional layers (optional).
    """

    def __init__(
        self, in_channels: int, out_channels: int, upsample_factor: int = 1, num_layers: int = 2
    ):
        super().__init__()
        layers = []
        current_channels = in_channels

        for _ in range(num_layers):
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*layers)
        self.upsample_factor = upsample_factor

        # Final layer to match desired output channels
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].

        Returns:
            torch.Tensor: Decoded output [B, out_channels, H * factor, W * factor].
        """
        x = self.conv_layers(x)

        if self.upsample_factor > 1:
            # Nearest-neighbor interpolation for upsampling (simple placeholder)
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="nearest")

        x = self.final_conv(x)
        return x
