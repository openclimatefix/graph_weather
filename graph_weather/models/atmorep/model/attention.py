import torch
import torch.nn as nn

from ..config import AtmoRepConfig


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism that enables communication between transformers of different fields.
    It allows a target tensor (x) to attend to a source tensor (context) and is useful in
    multi-modal tasks where information from different fields is combined.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters like hidden_dim,
                                 num_heads, and attention_dropout.
    """

    def __init__(self, config: AtmoRepConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim**-0.5

        # Projection layers for Q, K, V and output projection
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x, context):
        """
        Forward pass through the CrossAttention mechanism. It computes attention scores
        between the target tensor (x) and the context tensor, producing the attended output.

        Args:
            x (torch.Tensor): The target tensor, shape [B, T, N, D], where B is batch size,
                              T is the number of time steps, N is the number of spatial tokens,
                              and D is the hidden dimension.
            context (torch.Tensor): The source tensor, shape [B, T, N, D], typically from a different field.

        Returns:
            torch.Tensor: The attended output tensor, shape [B, T, N, D].
        """
        B, T, N, D = x.shape

        # Compute projections for Q, K, V
        q = self.q_proj(x).reshape(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = (
            self.k_proj(context)
            .reshape(B, T, N, self.num_heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )
        v = (
            self.v_proj(context)
            .reshape(B, T, N, self.num_heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )

        # Compute attention scores and apply softmax
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Compute the attended output
        out = (attn @ v).transpose(2, 3).reshape(B, T, N, D)
        out = self.out_proj(out)

        return out


class SpatioTemporalAttention(nn.Module):
    """
    Enhanced attention mechanism that explicitly models both spatial and temporal dependencies.
    It first applies spatial attention within each time step, then applies temporal attention
    for each spatial position. This approach is useful for tasks involving spatiotemporal data
    like videos or time-series data with spatial resolution.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters like hidden_dim,
                                 num_heads, and attention_dropout.
    """

    def __init__(self, config: AtmoRepConfig):
        super().__init__()
        self.config = config

        # Spatial attention using multi-head attention for each time step
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
        )

        # Temporal attention using multi-head attention for each spatial position
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
        )

        # Final output projection after the attention operations
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass through the SpatioTemporalAttention mechanism. It applies spatial attention
        to each time step, followed by temporal attention for each spatial position.

        Args:
            x (torch.Tensor): Input tensor, shape [B, T, N, D], where B is batch size,
                              T is the number of time steps, N is the number of spatial tokens,
                              and D is the hidden dimension.

        Returns:
            torch.Tensor: The output tensor after applying both spatial and temporal attention,
                          shape [B, T, N, D].
        """
        B, T, N, D = x.shape

        # Apply spatial attention within each time step
        spatial_out = []
        for t in range(T):
            # Transpose to match expected input format for MultiheadAttention [N, B, D]
            tokens = x[:, t].transpose(0, 1)
            attn_out, _ = self.spatial_attn(tokens, tokens, tokens)
            spatial_out.append(attn_out.transpose(0, 1))  # Return to [B, N, D]

        # Stack the results to [B, T, N, D]
        spatial_out = torch.stack(spatial_out, dim=1)

        # Apply temporal attention across all spatial positions
        temporal_out = []
        for n in range(N):
            # Transpose to [T, B, D] for temporal attention
            tokens = spatial_out[:, :, n].transpose(0, 1)
            attn_out, _ = self.temporal_attn(tokens, tokens, tokens)
            temporal_out.append(attn_out.transpose(0, 1))  # Return to [B, T, D]

        # Stack the results to [B, T, N, D]
        temporal_out = torch.stack(temporal_out, dim=2)

        # Final projection to output the attended features
        out = self.proj(temporal_out)
        out = self.dropout(out)

        return out
