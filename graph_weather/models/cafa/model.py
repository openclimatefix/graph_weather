"""CaFA model module for weather forecasting.

Contains the main CaFA model implementation combining encoder, processor, and decoder.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import CaFADecoder
from .encoder import CaFAEncoder
from .processor import CaFAProcessor


class CaFAForecaster(nn.Module):
    """CaFA (Climate-Aware Factorized Attention) model.

    Puts together Encoder, Processor and Decoder into an end-to-end model.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        model_dim: int = 256,
        downsampling_factor: int = 2,
        processor_depth: int = 6,
        num_heads: int = 8,
        dim_head: int = 64,
        feedforward_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        """Initialize the CaFA forecaster.
        
        Args:
            input_channels: No. of input channels/features
            output_channels: No. of channels to predict
            model_dim: Internal dimensions of the model
            downsampling_factor: Down/Up-sampling factor in the encoder-decoder
            processor_depth: No. of transformer blocks in the processor
            num_heads: No. of attention heads in each block
            dim_head: Dimension of each attention head
            feedforward_multiplier: Multiplier for the feedforward network's inner dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.downsampling_factor = downsampling_factor

        self.encoder = CaFAEncoder(
            input_channels=input_channels,
            model_dim=model_dim,
            downsampling_factor=downsampling_factor,
        )

        self.processor = CaFAProcessor(
            dim=model_dim,
            depth=processor_depth,
            heads=num_heads,
            dim_head=dim_head,
            feedforward_multiplier=feedforward_multiplier,
            dropout=dropout,
        )

        self.decoder = CaFADecoder(
            model_dim=model_dim,
            output_channels=output_channels,
            upsampling_factor=downsampling_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CaFA forecaster.
        
        Args:
            x: Input tensor of shape (batch, input_channels, height, width)

        Returns:
            Output tensor of shape (batch, output_channels, height, width)
        """

        # to handle odd-sized inputs, we pad the input to be divisible by downsampling factor
        _, _, h, w = x.shape
        pad_h = (
            self.downsampling_factor - (h % self.downsampling_factor)
        ) % self.downsampling_factor
        pad_w = (
            self.downsampling_factor - (w % self.downsampling_factor)
        ) % self.downsampling_factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.encoder(x)
        x = self.processor(x)
        x = self.decoder(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        return x
