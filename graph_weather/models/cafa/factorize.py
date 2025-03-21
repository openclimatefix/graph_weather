import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AxialAttention(nn.Module):
    """
    Axial Attention mechanism that performs attention separately along the height and width axes.

    This approach reduces the computational complexity from O((H*W)²) to O(H*W*(H+W)) by decomposing
    the 2D attention into two 1D attentions. The mechanism first computes attention along the height axis
    (each column attends to all rows) and then along the width axis (each row attends to all columns).

    Args:
        dim (int): Embedding dimension of the input.
        num_heads (int): Number of attention heads.
        dim_head (int, optional): Dimension of each attention head. Defaults to dim // num_heads.
        dropout (float, optional): Dropout probability applied after the linear projections. Default: 0.0.

    Expected Input:
        x (torch.Tensor): Input tensor of shape [B, N, C] where N = H * W, B is the batch size,
                          and C is the embedding dimension.
        h (int): Height of the feature map.
        w (int): Width of the feature map.

    Returns:
        torch.Tensor: Output tensor of shape [B, N, C] after applying axial attention.
    """

    def __init__(self, dim: int, num_heads: int = 4, dim_head: int = None, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head or (dim // num_heads)
        self.scale = self.dim_head**-0.5

        inner_dim = self.dim_head * num_heads

        # Projections for height-axis attention
        self.to_q_h = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_h = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_h = nn.Linear(dim, inner_dim, bias=False)

        # Projections for width-axis attention
        self.to_q_w = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_w = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_w = nn.Linear(dim, inner_dim, bias=False)

        # Output projections after each axis attention
        self.to_out_h = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.to_out_w = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Forward pass implementing the axial attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C] with N = H * W.
            h (int): Height of the feature map.
            w (int): Width of the feature map.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, C].
        """
        b, n, c = x.shape
        assert n == h * w, f"Input tokens {n} must equal h*w, but got {h}*{w}={h*w}"
        # Reshape to spatial layout [B, H, W, C]
        x_spatial = x.view(b, h, w, c)

        # ----- Height-axis attention -----
        # For height attention, process each column independently.
        # Rearrange to [B, heads, H, W, dim_head] for height axis.
        q_h = self.to_q_h(x_spatial)
        k_h = self.to_k_h(x_spatial)
        v_h = self.to_v_h(x_spatial)
        q_h = rearrange(q_h, "b h w (heads d) -> b heads h w d", heads=self.num_heads)
        k_h = rearrange(k_h, "b h w (heads d) -> b heads h w d", heads=self.num_heads)
        v_h = rearrange(v_h, "b h w (heads d) -> b heads h w d", heads=self.num_heads)
        # For each column (w), attend along the height dimension.
        q_h = rearrange(q_h, "b heads h w d -> (b heads w) h d")
        k_h = rearrange(k_h, "b heads h w d -> (b heads w) h d")
        v_h = rearrange(v_h, "b heads h w d -> (b heads w) h d")
        attn_h = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale  # [B*heads*w, h, h]
        attn_h = F.softmax(attn_h, dim=-1)
        out_h = torch.matmul(attn_h, v_h)  # [B*heads*w, h, d]
        # Restore shape to [B, H, W, inner_dim]
        out_h = rearrange(out_h, "(b heads w) h d -> b w heads h d", heads=self.num_heads, w=w)
        out_h = rearrange(out_h, "b w heads h d -> b h w (heads d)")
        out_h = self.to_out_h(out_h)

        # ----- Width-axis attention -----
        # For width attention, process each row independently.
        q_w = self.to_q_w(x_spatial)
        k_w = self.to_k_w(x_spatial)
        v_w = self.to_v_w(x_spatial)
        q_w = rearrange(q_w, "b h w (heads d) -> b heads h w d", heads=self.num_heads)
        k_w = rearrange(k_w, "b h w (heads d) -> b heads h w d", heads=self.num_heads)
        v_w = rearrange(v_w, "b h w (heads d) -> b heads h w d", heads=self.num_heads)
        # For each row (h), attend along the width dimension.
        q_w = rearrange(q_w, "b heads h w d -> (b heads h) w d")
        k_w = rearrange(k_w, "b heads h w d -> (b heads h) w d")
        v_w = rearrange(v_w, "b heads h w d -> (b heads h) w d")
        attn_w = torch.matmul(q_w, k_w.transpose(-2, -1)) * self.scale  # [B*heads*h, w, w]
        attn_w = F.softmax(attn_w, dim=-1)
        out_w = torch.matmul(attn_w, v_w)  # [B*heads*h, w, d]
        out_w = rearrange(out_w, "(b heads h) w d -> b h heads w d", heads=self.num_heads, h=h)
        out_w = rearrange(out_w, "b h heads w d -> b h w (heads d)")
        out_w = self.to_out_w(out_w)

        # Combine both axis attentions.
        out = out_h + out_w  # [B, H, W, C]
        # Flatten back to [B, N, C]
        out = out.view(b, h * w, c)
        return out


class FactorizedTransformerBlock(nn.Module):
    """
    Transformer block with Axial Attention and a feed-forward MLP.

    This enhanced block integrates the axial attention mechanism for efficient self-attention along
    separate spatial axes, and it optionally adds learnable positional encodings to retain spatial context.
    The design follows modern transformer architectures with pre-normalization and residual connections.

    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion factor for the hidden dimension in the MLP.
        dropout (float): Dropout probability applied in both attention and MLP layers.

    Expected Input:
        x (torch.Tensor): Input tensor of shape [B, N, dim], where N typically equals H * W.
        h (int): Height of the feature map.
        w (int): Width of the feature map.

    Returns:
        torch.Tensor: Output tensor of shape [B, N, dim].
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AxialAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network (MLP) with GELU non-linearity.
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

        # Optional learnable positional encoding to maintain spatial layout.
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            # Assumes a maximum of 1024 tokens (e.g., a 32x32 spatial grid).
            self.pos_encoding = nn.Parameter(torch.zeros(1, 1024, dim))
            nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Forward pass of the transformer block with axial attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, dim] with N = H * W.
            h (int): Height of the feature map.
            w (int): Width of the feature map.

        Returns:
            torch.Tensor: Output tensor of shape [B, N, dim].
        """
        b, n, c = x.shape

        # Optionally add positional encoding.
        if self.use_pos_encoding:
            pos_enc = self.pos_encoding[:, :n, :]
            x = x + pos_enc

        # Apply self-attention with pre-normalization and add residual connection.
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, h, w)

        # Apply the feed-forward network with pre-normalization and residual connection.
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x


class FactorizedAttention(nn.Module):
    """
    Legacy Factorized Attention mechanism.

    This implementation provides a simpler, non-axial factorized attention approach maintained
    for backward compatibility and benchmarking. It uses standard multi-head attention by linearly projecting
    the input into queries, keys, and values, and then computing attention using scaled dot-product.

    Args:
        dim (int): Embedding dimension of the input.
        num_heads (int): Number of attention heads.

    Expected Input:
        x (torch.Tensor): Input tensor of shape [B, N, dim] where N is the flattened spatial dimension
            (e.g., H * W for a 2D feature map).

    Returns:
        torch.Tensor: Output tensor of shape [B, N, dim].
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Linear projections for queries, keys, and values.
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Split projections into multiple heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        # Compute scaled dot-product attention.
        attn_scores = torch.einsum("b n h d, b m h d -> b h n m", q, k) / (D**0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        out = torch.einsum("b h n m, b m h d -> b n h d", attn_weights, v)

        # Merge attention heads.
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.out_proj(out)
        return out
