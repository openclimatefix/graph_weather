import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism that enables communication between different embeddings.
    A target tensor (x) attends to a source tensor (context), which is useful in
    multi-modal tasks or whenever you want one representation to attend to another.

    Args:
        hidden_dim (int): Dimensionality of the hidden embeddings.
        num_heads (int): Number of attention heads.
        attention_dropout (float, optional): Dropout probability applied to attention weights.
            Defaults to 0.1.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers for Q, K, V and the output projection
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CrossAttention mechanism. Computes attention scores
        between the target tensor (x) and the source tensor (context), producing
        the attended output.

        Args:
            x (torch.Tensor): Target tensor of shape [B, T, N, D], where:
                - B is the batch size
                - T is the number of time steps
                - N is the number of tokens (e.g., spatial positions)
                - D is the hidden dimension
            context (torch.Tensor): Source tensor of the same shape as x, or
                [B, T, N, D].

        Returns:
            torch.Tensor: The attended output tensor of shape [B, T, N, D].
        """
        B, T, N, D = x.shape

        # Compute projections for Q, K, V
        # Reshape to [B, T, N, num_heads, head_dim], then permute for attention
        q = self.q_proj(x).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(context).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(context).view(B, T, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Compute attention scores: [B, T, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Compute the attended output: [B, T, num_heads, N, head_dim]
        out = attn @ v
        # Reshape back to [B, T, N, D]
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, N, D)

        # Final linear projection
        out = self.out_proj(out)

        return out


class SpatioTemporalAttention(nn.Module):
    """
    Attention mechanism that explicitly models both spatial and temporal dependencies.
    It first applies spatial attention within each time step, then applies temporal
    attention across time for each spatial position.

    Args:
        hidden_dim (int): Dimensionality of the hidden embeddings.
        num_heads (int): Number of attention heads.
        attention_dropout (float, optional): Dropout probability for the attention layers.
            Defaults to 0.1.
        dropout (float, optional): Dropout probability applied to the final output.
            Defaults to 0.1.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Spatial attention for each time step
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,  # to simplify shape handling
        )

        # Temporal attention for each spatial position
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Final projection and dropout
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SpatioTemporalAttention mechanism. It applies:
          1. Spatial attention to each time step.
          2. Temporal attention across time for each spatial position.

        Args:
            x (torch.Tensor): Input of shape [B, T, N, D], where:
                - B is batch size
                - T is number of time steps
                - N is number of spatial tokens
                - D is hidden dimension

        Returns:
            torch.Tensor: The output tensor after applying spatial and temporal attention,
                          shape [B, T, N, D].
        """
        B, T, N, D = x.shape

        # === 1) Spatial Attention within each time step ===
        # We'll process each time slice [B, N, D] with MultiheadAttention.
        # MultiheadAttention expects [B, seq_len, embed_dim], so we treat N as seq_len.

        # We'll collect each time step's output in a list.
        spatial_outputs = []
        for t in range(T):
            # x[:, t] => [B, N, D]
            # Reshape to [B, N, D] => already the correct shape for batch_first
            spatial_out, _ = self.spatial_attn(x[:, t], x[:, t], x[:, t])
            spatial_outputs.append(spatial_out)

        # Stack along the time dimension => [B, T, N, D]
        spatial_outputs = torch.stack(spatial_outputs, dim=1)

        # === 2) Temporal Attention across each spatial position ===
        # Now we treat T as seq_len, for each spatial position N.
        temporal_outputs = []
        for n in range(N):
            # spatial_outputs[:, :, n] => [B, T, D]
            # This is the sequence for time steps at spatial index n
            tmp_out, _ = self.temporal_attn(
                spatial_outputs[:, :, n],  # query
                spatial_outputs[:, :, n],  # key
                spatial_outputs[:, :, n],  # value
            )
            temporal_outputs.append(tmp_out)

        # temporal_outputs is a list of [B, T, D], one per spatial position => [N]
        # Stack => [N, B, T, D], then permute => [B, T, N, D]
        temporal_outputs = torch.stack(temporal_outputs, dim=2)

        # === Final projection and dropout ===
        out = self.proj(temporal_outputs)
        out = self.dropout(out)

        return out