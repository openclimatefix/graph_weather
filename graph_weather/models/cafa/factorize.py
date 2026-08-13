"""
Core components for the Factorized Attention mechanism.

Based on the principles of Axial Attention.
"""

from einops import rearrange
from torch import einsum, nn


def FeedFoward(dim, multiply=4, dropout=0.0):
    """
    Standard feed-forward block used in transformer architecture.

    Consists of 2 linear layers with GELU activation and dropouts, in between.

    Args:
        dim: Feature dimension of the input and of the output.
        multiply: Multiplier for the hidden dimension of the inner linear layer.
        dropout: Dropout probability applied after the activation and after the projection.

    Returns:
        An ``nn.Sequential`` module mapping features of size ``dim`` back to size ``dim``.
    """
    inner_dim = int(dim * multiply)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout),
    )


class AxialAttention(nn.Module):
    """
    Performs multi-head self-attention on a single axis of a 2D feature map.

    Core building block for Factorized Attention.
    """

    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        """
        Initialize the axial attention block.

        Args:
            dim: Feature dimension of the input and of the output.
            heads: Number of attention heads.
            dim_head: Feature dimension of each attention head.
            dropout: Dropout probability applied to the attention weights.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, axis):
        """
        Forward pass for axial attention.

        Args:
            x: Input tensor of shape (batch, height, width, channels)
            axis: Axis to perform attention on (1 for height, 2 for width)

        Returns:
            Tensor of shape (batch, height, width, channels), the same shape as ``x``.

        Raises:
            ValueError: If ``axis`` is neither 1 nor 2.
        """
        b, h, w, d = x.shape

        # rearrange tensor to isolate attention axis as the sequence dim
        if axis == 1:
            x = rearrange(x, "b h w d -> (b w) h d")  # attention along height
        elif axis == 2:
            x = rearrange(x, "b h w d -> (b h) w d")  # attention along width
        else:
            raise ValueError("Axis must be 1 (height) or 2 (width)")

        # project to query, key and value tensors
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )  # reshape for multi-head attn

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale  # attention scores
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # attn to the value tensors
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # original 2D grid format
        if axis == 1:
            out = rearrange(out, "(b w) h d -> b h w d", w=w)
        elif axis == 2:
            out = rearrange(out, "(b h) w d -> b h w d", h=h)

        return out


class FactorizedAttention(nn.Module):
    """
    Combines 2 AxialAttention blocks to perform full factorized attention.

    The blocks are applied over a 2D feature map, first along height then along width.
    """

    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        """
        Initialize the factorized attention block.

        Args:
            dim: Feature dimension of the input and of the output.
            heads: Number of attention heads used by each axial attention block.
            dim_head: Feature dimension of each attention head.
            dropout: Dropout probability applied to the attention weights.
        """
        super().__init__()
        self.attn_height = AxialAttention(dim, heads, dim_head, dropout)
        self.attn_width = AxialAttention(dim, heads, dim_head, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Attend along the height axis, then along the width axis.

        Each axial attention block is applied with a pre-norm residual connection.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Tensor of the same shape as ``x``.
        """
        x = x + self.attn_height(self.norm1(x), axis=1)
        x = x + self.attn_width(self.norm2(x), axis=2)
        return x


class FactorizedTransformerBlock(nn.Module):
    """
    Standalone transformer block using Factorized attention.

    Pairs a factorized attention sub-layer with a feed-forward sub-layer, each
    applied with a pre-norm residual connection.
    """

    def __init__(self, dim, heads, dim_head=64, feedforward_multiplier=4, dropout=0.0):
        """
        Initialize the factorized transformer block.

        Args:
            dim: Feature dimension of the input and of the output.
            heads: Number of attention heads used by each axial attention block.
            dim_head: Feature dimension of each attention head.
            feedforward_multiplier: Multiplier for the hidden dimension of the feed-forward block.
            dropout: Dropout probability used by the attention and feed-forward blocks.
        """
        super().__init__()
        self.attn = FactorizedAttention(dim, heads, dim_head, dropout)
        self.ffn = FeedFoward(dim, feedforward_multiplier, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Apply factorized attention and the feed-forward block, both with residuals.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Tensor of the same shape as ``x``.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
