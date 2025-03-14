import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
      1. Layer Normalization
      2. Multi-Head Attention
      3. Residual Connection
      4. Layer Normalization
      5. MLP (feed-forward) Layer
      6. Residual Connection

    Args:
        hidden_dim (int): The hidden dimension of the input/output tokens.
        num_heads (int): The number of attention heads in the multi-head attention module.
        mlp_ratio (float): Multiplier for the hidden dimension of the MLP. If `mlp_ratio=4.0`,
            then the MLP hidden size is `4 * hidden_dim`.
        dropout (float, optional): Dropout probability used in the MLP layers. Defaults to 0.0.
        attention_dropout (float, optional): Dropout probability applied to the attention weights.
            Defaults to 0.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(
            dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C], where:
                - B is the batch size
                - N is the sequence length (number of tokens)
                - C is the hidden dimension (channels)

        Returns:
            torch.Tensor: Output tensor of the same shape [B, N, C].
        """
        # Multi-head attention + residual
        x = x + self.attn(self.norm1(x))
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism that computes attention weights and applies them to the input.

    Args:
        dim (int): The dimensionality of the input and output.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate to apply on attention weights and final projection.
            Defaults to 0.0.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Linear layers to compute Q, K, V
        self.qkv = nn.Linear(dim, dim * 3)
        # Projection back to original dimension
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C], where:
                - B is the batch size
                - N is the sequence length
                - C is the hidden dimension

        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism, shape [B, N, C].
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Permute => Q, K, V each of shape [B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate attention weights => shape [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention => shape [B, num_heads, N, head_dim]
        x = attn @ v
        # Rearrange back => shape [B, N, num_heads, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    A simple MLP (feed-forward) network used inside Transformer blocks.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of features in the hidden layer.
        out_features (int): Number of output features.
        dropout (float, optional): Dropout probability to apply after each layer. Defaults to 0.0.
    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C].

        Returns:
            torch.Tensor: Output tensor of shape [B, N, out_features].
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
