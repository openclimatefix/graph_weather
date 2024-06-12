"""Modules"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


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
