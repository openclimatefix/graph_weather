"""Experimental sparse transformer using DGL sparse."""

import dgl.sparse as dglsp
import torch
import torch.nn as nn

from graph_weather.models.gencast.layers.modules import ConditionalLayerNorm


class SparseAttention(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, input_dim=512, output_dim=512, num_heads=4):
        """Initialize Sparse MultiHead attention module.

        Args:
            input_dim (int): input dimension. Defaults to 512.
            output_dim (int): output dimension. Defaults to 512.
            num_heads (int): number of heads. Output dimension should be divisible by num_heads.
                Defaults to 4.
        """
        super().__init__()
        if output_dim % num_heads:
            raise ValueError("Output dimension should be divisible by the number of heads.")

        self.hidden_size = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: dglsp.SparseMatrix):
        """Forward pass of SparseMHA.

        Args:
            x (torch.Tensor): input tensor.
            adj (SparseMatrix): adjacency matrix in DGL SparseMatrix format.

        Returns:
            y (tensor): output of MultiHead attention.
        """
        N = len(x)
        # computing query,key and values.
        q = self.q_proj(x).reshape(N, self.head_dim, self.num_heads)  # (dense) [N, dh, nh]
        k = self.k_proj(x).reshape(N, self.head_dim, self.num_heads)  # (dense) [N, dh, nh]
        v = self.v_proj(x).reshape(N, self.head_dim, self.num_heads)  # (dense) [N, dh, nh]
        # scaling query
        q *= self.scaling

        # sparse-dense-dense product
        attn = dglsp.bsddmm(adj, q, k.transpose(1, 0))  # (sparse) [N, N, nh]

        # sparse softmax (by default applies on the last sparse dimension).
        attn = attn.softmax()  # (sparse) [N, N, nh]

        # sparse-dense multiplication
        out = dglsp.bspmm(attn, v)  # (dense) [N, dh, nh]
        return self.out_proj(out.reshape(N, -1))


class SparseTransformer(nn.Module):
    """A single transformer block for graph neural networks.

    This module implements a single transformer block with a sparse attention mechanism.
    """

    def __init__(
        self,
        conditioning_dim: int,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation_layer: torch.nn.Module = nn.ReLU,
    ):
        """Initialize SparseTransformer module.

        Args:
            conditioning_dim (int, optional): dimension of the conditioning parameter. If None the
                layer normalization will not be applied.
            input_dim (int): dimension of the input features.
            output_dim (int): dimension of the output features.
            edges_dim (int): dimension of the edge features.
            num_heads (int): number of heads for multi-head attention.
            activation_layer (torch.nn.Module): activation function applied before
                returning the output.
        """
        super().__init__()

        # initialize multihead sparse attention.
        self.sparse_attention = SparseAttention(
            input_dim=input_dim, output_dim=output_dim, num_heads=num_heads
        )

        # initialize mlp
        self.activation = activation_layer()
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim), self.activation, nn.Linear(output_dim, output_dim)
        )

        # initialize conditional layer normalization
        self.cond_norm_1 = ConditionalLayerNorm(
            conditioning_dim=conditioning_dim, features_dim=output_dim
        )
        self.cond_norm_2 = ConditionalLayerNorm(
            conditioning_dim=conditioning_dim, features_dim=output_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        cond_param: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Apply SparseTransformer to input.

        Input and conditioning parameter must have same batch size.

        Args:
            x (torch.Tensor): tensor containing nodes features.
            edge_index (torch.Tensor): edge index tensor.
            cond_param (torch.Tensor): conditioning parameter.
            *args: ignored by the module.
            **kwargs: ignored by the module.

        """
        x = x + self.sparse_attention.forward(
            x=x, adj=dglsp.spmatrix(indices=edge_index, shape=(x.shape[0], x.shape[0]))
        )
        x = self.cond_norm_1(x, cond_param)
        x = x + self.mlp(x)
        x = self.cond_norm_2(x, cond_param)
        return x
