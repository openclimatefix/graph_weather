import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder
from .processor import Processor

class CaFA(nn.Module):
    """
    A generic CaFA-like model with an Encoder, a Processor (Factorized Transformer),
    and a Decoder. The architecture is designed to be flexible and can be adapted
    to different weather variables or tasks.

    Args:
        in_channels (int): Number of input channels (e.g., surface + upper-air vars).
        hidden_dim (int): Dimension of features in encoder and processor.
        out_channels (int): Number of output channels (e.g., next-step predictions).
        encoder_downsample (int): Downsample factor in the encoder.
        decoder_upsample (int): Upsample factor in the decoder.
        processor_depth (int): Number of Transformer blocks in the processor.
        num_heads (int): Number of attention heads in the processor.
        mlp_ratio (float): Expansion ratio for the MLP in Transformer blocks.
        dropout (float): Dropout probability in the processor.
        encoder_layers (int): Number of convolutional layers in the encoder.
        decoder_layers (int): Number of convolutional layers in the decoder.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 out_channels: int,
                 encoder_downsample: int = 1,
                 decoder_upsample: int = 1,
                 processor_depth: int = 4,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 encoder_layers: int = 2,
                 decoder_layers: int = 2):
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            downsample_factor=encoder_downsample,
            num_layers=encoder_layers
        )
        
        self.processor = Processor(
            dim=hidden_dim,
            depth=processor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            in_channels=hidden_dim,
            out_channels=out_channels,
            upsample_factor=decoder_upsample,
            num_layers=decoder_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CaFA-like model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H*, W*],
                          where H* and W* depend on the upsample factor.
        """
        # 1. Encode
        x_enc = self.encoder(x)
        
        # 2. Process
        B, C, H, W = x_enc.shape
        x_proc = self.processor(x_enc, H, W)
        
        # 3. Decode
        x_dec = self.decoder(x_proc)
        
        return x_dec
