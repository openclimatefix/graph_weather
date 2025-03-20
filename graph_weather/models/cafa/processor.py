import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .factorize import FactorizedTransformerBlock

class Processor(nn.Module):
    """
    Processor module that applies multiple FactorizedTransformerBlocks.

    Args:
        dim (int): Embedding dimension.
        depth (int): Number of Transformer blocks to stack.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion factor for the MLP.
        dropout (float): Dropout probability.
    """
    def __init__(self,
                 dim: int,
                 depth: int = 4,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            FactorizedTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass of the processor.

        Args:
            x (torch.Tensor): [B, C, H, W] feature map where.
            - B: Batch size.
            - C: Number of channels (should equal the embedding dimension).
            - H: Height of the feature map(needed for reshaping).
            - W: Width of the feature map(needed for reshaping).

        Returns:
            torch.Tensor: Processed features of shape [B, C, H, W].
        """
        B, C, _, _ = x.shape
        
        # Flatten spatial dimensions for factorized attention
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, H*W, C]
        
        for block in self.blocks:
            x = block(x)
        
        # Reshape back to [B, C, H, W]
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

