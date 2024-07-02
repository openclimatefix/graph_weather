"""Modules"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import TransformerConv


class MLP(nn.Module):
    """Classic multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation_layer: nn.Module = nn.ReLU,
        use_layer_norm: bool = False,
        bias: bool = True,
        activate_final: bool = False,
    ):
        """Initialize MLP module.

        Args:
            input_dim (int): dimension of input.
            hidden_dims (List[int]): list of hidden linear layers dimensions.
            activation_layer (torch.nn.Module): activation
                function to use. Defaults to torch.nn.ReLU.
            use_layer_norm (bool, optional): if Ttrue apply LayerNorm to output. Defaults to False.
            bias (bool, optional): if true use bias in linear layers. Defaults to True.
            activate_final (bool, optional): whether to apply  the activation function to the final
                layer. Defaults to False.
        """
        super().__init__()

        # Initialize linear layers
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dims[0], bias=bias))
        for i in range(0, len(hidden_dims) - 1):
            self.linears.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias))

        # Initialize activation
        self.activation = activation_layer()

        # Initialize layer normalization
        self.norm_layer = None
        if use_layer_norm:
            self.norm_layer = nn.LayerNorm(hidden_dims[-1])

        self.activate_final = activate_final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input."""

        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.activation(x)

        x = self.linears[-1](x)

        if self.activate_final:
            x = self.activation(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class InteractionNetwork(MessagePassing):
    """Single message-passing interaction network as described in GenCast.

    This network performs two steps:
    1) message-passing: e'_ij = MLP([e_ij,v_i,v_j])
    2) aggregation: v'_j = MLP([v_j, sum_i {e'_ij}])
    The underlying graph is a directed graph.

    Note:
        We don't need to update edges in GenCast, hence we skip it.
    """

    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        edge_attr_dim: int,
        hidden_dims: list[int],
        use_layer_norm: bool = False,
        activation_layer: nn.Module = nn.ReLU,
    ):
        """Initialize the Interaction Network.

        Args:
            sender_dim (int): dimension of sender nodes' features.
            receiver_dim (int): dimension of receiver nodes' features.
            edge_attr_dim (int): dimension of the edge features.
            hidden_dims (list[int]): list of sizes of MLP's linear layers.
            use_layer_norm (bool): if true add layer normalization to MLP's last layer.
                Defaults to False.
            activation_layer (torch.nn.Module): activation function. Defaults to nn.ReLU.
        """
        super().__init__(aggr="add", flow="source_to_target")
        self.mlp_edges = MLP(
            input_dim=sender_dim + receiver_dim + edge_attr_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )
        self.mlp_nodes = MLP(
            input_dim=receiver_dim + hidden_dims[-1],
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

    def message(self, x_i, x_j, edge_attr):
        """Message-passing step."""
        x = torch.cat((x_i, x_j, edge_attr), dim=-1)
        x = self.mlp_edges(x)
        return x

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the output of the Interaction Network.

        This method processes the input node features and edge attributes through
        the network to produce the output node features.

        Args:
            x (tuple[torch.Tensor, torch.Tensor]): a tuple containing:
                - sender nodes' features (torch.Tensor): features of the sender nodes.
                - receiver nodes' features (torch.Tensor): features of the receiver nodes.
            edge_index (torch.Tensor): tensor containing edge indices, defining the
                connections between nodes.
            edge_attr (torch.Tensor): tensor containing edge features, representing
                the attributes of each edge.

        Returns:
            torch.Tensor: the resulting node features after applying the Interaction Network.
        """
        aggr = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x[0].shape[0], x[1].shape[0])
        )
        out = self.mlp_nodes(torch.cat((x[1], aggr), dim=-1))
        return out


class FourierEmbedding(nn.Module):
    """Fourier embedding module."""

    def __init__(self, output_dim: int, num_frequencies: int, base_period: int):
        """Initialize the Fourier Embedder.

        Args:
            output_dim (int): output dimension.
            num_frequencies (int): number of frequencies for sin/cos embedding.
            base_period (int): max period for sin/cos embedding.
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            nn.Linear(2 * num_frequencies, output_dim, bias=True),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim, bias=True),
        )
        self.num_frequencies = num_frequencies
        self.base_period = base_period

    def fourier_features(self, t):
        """
        Create sinusoidal embeddings.
        """
        freqs = torch.exp(
            -math.log(self.base_period)
            * torch.arange(start=0, end=self.num_frequencies, dtype=torch.float32)
            / self.num_frequencies
        ).to(device=t.device)
        args = t * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        """Apply fourier features and mlp to the input."""
        t = self.fourier_features(t)
        t = self.mlp(t)
        return t


class ConditionalLayerNorm(nn.Module):
    """Conditional Layer Normalization.

    This module is a variant of Layer Normalization: an elementwise affine transformation is applied
    to the output of the LayerNorm, with parameters computed as Linears of some conditioning params.
    """

    def __init__(
        self,
        conditioning_dim: int,
        features_dim: int,
    ):
        """Initialize Conditional Layer Normalization module.

        Args:
            conditioning_dim (int): dimension of the conditioning parameter.
            features_dim (int): dimension of the input features.
        """
        super().__init__()

        # Initialize linear layers
        self.linear_scale = nn.Linear(conditioning_dim, features_dim, bias=True)
        self.linear_bias = nn.Linear(conditioning_dim, features_dim, bias=True)

        # Initialize the LayerNorm
        self.norm = nn.LayerNorm(features_dim, eps=1e-05, elementwise_affine=False)

    def forward(self, x: torch.Tensor, cond_param: torch.Tensor) -> torch.Tensor:
        """Apply ConditionalLayerNorm to input.

        Input and conditioning parameter must have same batch size.

        Args:
            x (torch.Tensor): input.
            cond_param (torch.Tensor): conditioning parameter.
        """

        # check shapes.
        if not x.shape[0] == cond_param.shape[0]:
            raise ValueError(
                "Expected same batch dimension for the input and the conditioning "
                f"parameter, got {x.shape[0]} and {cond_param.shape[0]}"
            )

        # compute parameters and normalize.
        scale = self.linear_scale(cond_param)
        bias = self.linear_bias(cond_param)
        x_norm = self.norm(x)

        assert scale.shape[1] == x_norm.shape[1]

        # apply elementwise affine transformation.
        out = scale * x_norm + bias
        return out


class CondTransformerBlock(nn.Module):
    """
    A single transformer block for graph neural networks.

    This module implements a transformer block tailored for graph structures, following the
    methodology outlined in the paper "Masked Label Prediction: Unified Message Passing Model
    for Semi-Supervised Classification." The implementation aligns with the principles described
    in this paper, providing an adaptation of the transformer's attention mechanism to
    graph data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        conditioning_dim: int | None = None,
        edges_dim: int | None = None,
        concat: bool = True,
        beta: bool = True,
        activation_layer: torch.nn.Module | None = nn.ReLU,
    ):
        """Initialize Conditional Layer Normalization module.

        Args:
            input_dim (int): dimension of the input features.
            output_dim (int): dimension of the output features.
            edges_dim (int): dimension of the edge features.
            num_heads (int): number of heads for multi-head attention.
            conditioning_dim (int, optional): dimension of the conditioning parameter. If None the
                layer normalization will not be applied.
            edges_dim (int, optional): dimension of the edges features. If None edges features will
                not be used inside TransformerConv.
            concat (bool): if true concatenate the outputs of each head, otherwise average them.
                Defaults to True.
            beta (bool): if true apply the beta weighting described in the paper. Defauls to True.
            activation_layer (torch.nn.Module, optional): activation function applied before
                returning the output. If None skip the activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        # Initialize layers
        self.transformer_conv = TransformerConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=num_heads,
            concat=concat,
            beta=beta,
            edge_dim=edges_dim,
        )

        self.activation = activation_layer() if activation_layer is not None else None

        if conditioning_dim is not None:
            final_dim = num_heads * output_dim if concat else output_dim
            self.cond_norm = ConditionalLayerNorm(
                conditioning_dim=conditioning_dim, features_dim=final_dim
            )
        else:
            self.cond_norm = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        cond_param: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply CondTransformerBlock to input.

        Input and conditioning parameter must have same batch size.

        Args:
            x (torch.Tensor): tensor containing nodes features.
            edge_index (torch.Tensor): edge index tensor.
            edge_attr (torch.Tensor, optional): tensor containing edges features.
            cond_param (torch.Tensor, optional): conditioning parameter.

        """
        x = self.transformer_conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if self.cond_norm is not None:
            x = self.cond_norm(x, cond_param)

        if self.activation is not None:
            x = self.activation(x)

        return x
