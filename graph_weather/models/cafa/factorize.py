import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FactorizedAttention(nn.Module):
    """
    An illustrative factorized (axial) attention mechanism.
    Real implementations might split attention along height and width
    separately, or implement a more sophisticated factorization scheme.

    Args:
        dim (int): Embedding dimension of the input.
        num_heads (int): Number of attention heads.

    Expected Input:
        x (torch.Tensor): Input features of shape [B, N, dim],
            where:
              - B is the batch size.
              - N is the number of tokens, typically representing the flattened spatial
                dimensions (e.g., N = H * W for a 2D feature map).
              - dim is the embedding dimension per token.
    
    Returns:
        torch.Tensor: Output features of shape [B, N, dim].
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Linear projections for queries, keys, values
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Split into multiple heads using einops for clarity
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.num_heads)

        # Factorized attention approach (placeholder)
        attn_scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / (D ** 0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bmhd->bnhd', attn_weights, v)
        
        # Merge heads
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.out_proj(out)
        
        return out
class FactorizedTransformerBlock(nn.Module):
    """
    A single Transformer block with FactorizedAttention and a feed-forward MLP.

    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion factor in the MLP layer.
        dropout (float): Dropout probability.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FactorizedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # Simple feed-forward MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.
        
        Args:
            x (torch.Tensor): [B, N, dim] input tensor.
        
        Returns:
            torch.Tensor: [B, N, dim] output tensor.
        """
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        
        # Feed-forward
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        
        return x