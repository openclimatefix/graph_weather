import torch
import torch.nn as nn
from einops import rearrange

from .factorize import FactorizedTransformerBlock


class Processor(nn.Module):
    """
    Processor module that applies multiple FactorizedTransformerBlocks.

    This module is designed to process a 2D feature map by flattening it, applying a series of
    transformer blocks that utilize efficient axial attention, and then reshaping it back to its
    original spatial dimensions.

    Args:
        dim (int): Embedding dimension (number of channels) of the input.
        depth (int): Number of Transformer blocks to stack.
        num_heads (int): Number of attention heads used in each Transformer block.
        mlp_ratio (float): Expansion factor for the hidden dimension in the feed-forward MLP.
        dropout (float): Dropout probability applied in both the attention and MLP layers.

    Expected Input:
        x (torch.Tensor): Input tensor of shape [B, C, H, W], where:
            - B: Batch size.
            - C: Number of channels (should match the embedding dimension).
            - H: Height of the feature map.
            - W: Width of the feature map.

    Returns:
        torch.Tensor: Processed features of shape [B, C, H, W].
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                FactorizedTransformerBlock(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass of the Processor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            H (int): Height of the feature map (needed for reshaping).
            W (int): Width of the feature map (needed for reshaping).

        Returns:
            torch.Tensor: Processed feature map of shape [B, C, H, W].
        """
        B, C, _, _ = x.shape

        # Flatten the spatial dimensions for transformer processing.
        x = rearrange(x, "b c h w -> b (h w) c")  # [B, H*W, C]

        # Process the flattened features through each transformer block.
        for block in self.blocks:
            x = block(x, H, W)

        # Reshape back to the original 2D feature map dimensions.
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x
