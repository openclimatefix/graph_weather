import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import AtmoRepConfig

class TransformerBlock(nn.Module):
    """
    A transformer block consisting of multi-head attention and a feed-forward MLP layer. 
    The transformer block applies Layer Normalization, Multi-Head Attention, and an MLP to the input.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters such as 
                                 hidden dimension, number of attention heads, etc.
    """
    def __init__(self, config: AtmoRepConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(
            config.hidden_dim, 
            config.num_heads, 
            dropout=config.attention_dropout
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(
            config.hidden_dim, 
            config.hidden_dim * config.mlp_ratio, 
            config.hidden_dim, 
            dropout=config.dropout
        )
        
    def forward(self, x):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C], where B is batch size, 
                              N is sequence length, and C is the channel dimension.
        
        Returns:
            torch.Tensor: The output tensor after applying multi-head attention and MLP.
        """
        # Apply multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # Apply MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism that computes attention weights and applies them to the input.

    Args:
        dim (int): The dimensionality of the input and output.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate to apply on attention weights. Default is 0.0.
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)  # Linear layer to compute queries, keys, and values
        self.proj = nn.Linear(dim, dim)  # Linear layer to project the output back to the original dimension
        self.dropout = nn.Dropout(dropout)  # Dropout layer to regularize the attention weights
        
    def forward(self, x):
        """
        Forward pass of the MultiHeadAttention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C], where B is batch size, 
                              N is the number of tokens, and C is the channel dimension.
        
        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism.
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # Softmax to get attention weights
        attn = self.dropout(attn)  # Apply dropout to attention weights
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Project the output back to the original dimension
        x = self.proj(x)
        x = self.dropout(x)  # Apply dropout to the output
        
        return x


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) consisting of two fully connected layers with GELU activations.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of features in the hidden layer.
        out_features (int): Number of output features.
        dropout (float, optional): Dropout rate to apply after each layer. Default is 0.0.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)  # First fully connected layer
        self.act = nn.GELU()  # GELU activation function
        self.fc2 = nn.Linear(hidden_features, out_features)  # Second fully connected layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        
    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C], where B is batch size, 
                              N is the number of tokens, and C is the number of channels/features.
        
        Returns:
            torch.Tensor: The output tensor after applying the MLP layers.
        """
        x = self.fc1(x)  # Apply first fully connected layer
        x = self.act(x)  # Apply GELU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply second fully connected layer
        x = self.dropout(x)  # Apply dropout again
        
        return x