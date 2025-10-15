import torch
from torch import nn

class CaFADecoder(nn.Module):
    """
    Decoder for for CaFA
    After the Processor and FactorizedTransformer generated a prediction 
    in the latent space, the decoder's role is to translate this abstract
    representation back into a physical prediction
    """
    def __init__(self,
                 model_dim: int,
                 output_channels: int,
                 upsampling_factor: int = 1
    ):
        """
        Args:
            output_channels: No. of channels/features in output prediction
            model_dim: Dimensions of the model's hidden layers (output channels)
            upsampling_factor: Factor to upsample the spatial dimensions. 
                Must match the downsampling factor in encoder.
        """
        super().__init__()
        self.decoder = nn.ConvTranspose2d(
            in_channels=model_dim,
            out_channels=output_channels,
            kernel_size=upsampling_factor,
            stride=upsampling_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, model_dim, height, width).
        Returns:
            Output tensor of shape (batch, output_channels, height*factor, width*factor)
        """
        x = self.decoder(x)
        return x