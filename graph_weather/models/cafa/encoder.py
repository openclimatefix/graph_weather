import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    A generic encoder that applies downsampling or any initial transformations
    to the input data. For weather data, this might include:
      - Convolutional layers
      - Positional encoding (if desired)
      - Simple downsampling to reduce spatial resolution

    Args:
        in_channels (int): Number of input channels (e.g., surface + upper-air variables).
        hidden_dim (int): The base hidden dimension for the encoder layers.
        downsample_factor (int): Factor by which to downsample spatial dimensions.
        num_layers (int): Number of convolutional layers (optional).
    """
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 downsample_factor: int = 1,
                 num_layers: int = 2):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        for _ in range(num_layers):
            layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = hidden_dim

        self.conv_layers = nn.Sequential(*layers)
        self.downsample_factor = downsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
        
        Returns:
            torch.Tensor: Encoded features of shape [B, hidden_dim, H//factor, W//factor].
        """
        x = self.conv_layers(x)
        
        if self.downsample_factor > 1:
            # Downsample spatially (e.g., average pooling)
            x = F.avg_pool2d(x, kernel_size=self.downsample_factor)
        
        return x