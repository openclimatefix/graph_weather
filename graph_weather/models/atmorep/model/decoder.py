from typing import List

import torch
import torch.nn as nn
from .transformer import TransformerBlock


class Decoder(nn.Module):
    """
    A Decoder module designed to process features of a given field,
    using transformer blocks with U-Net-like skip connections for efficient
    multi-resolution feature propagation.

    Args:
        hidden_dim (int): Dimensionality of the hidden embeddings.
        num_layers (int): Number of layers (transformer blocks) in the decoder.
        num_heads (int): Number of attention heads for each transformer block.
        dropout (float, optional): Dropout probability applied within transformer blocks.
            Defaults to 0.1.
        attention_dropout (float, optional): Dropout probability for attention weights.
            Defaults to 0.1.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        #  creating half as many blocks as num_layers // 2 to match the U-Net-like pattern
        self.blocks = nn.ModuleList(
            [
                # Use the default transformer block from the codebase
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers // 2)
            ]
        )

        # Skip connections: a linear projection for each block
        self.skip_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(self.blocks))]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the Decoder. Applies transformer blocks and uses skip
        connections from lower-resolution features (similar to a U-Net architecture).

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N, D], where:
                - B is batch size
                - T is the number of time steps
                - N is the number of patches/spatial tokens
                - D is the hidden dimension
            skip_features (List[torch.Tensor]): A list of skip-connection tensors from earlier
                layers, each shaped [B, T, N, D]. The list is typically in order from shallowest
                to deepest features. We'll pick from the end of this list for the most
                relevant skip.

        Returns:
            torch.Tensor: Output tensor of shape [B, T, N, D] after applying transformer
                          blocks and layer normalization.
        """
        # Process with transformer blocks in a U-Net-like pattern
        for i, block in enumerate(self.blocks):
            if i < len(skip_features):
                # Use skip features from the end of the list
                skip_idx = len(skip_features) - 1 - i
                # Project skip feature to match hidden_dim
                skip_proj = self.skip_projections[i](skip_features[skip_idx])
                # Combine skip feature with current input
                x = block(x + skip_proj)
            else:
                # If no skip feature available, just apply the block
                x = block(x)

        # Final normalization
        x = self.norm(x)
        return x