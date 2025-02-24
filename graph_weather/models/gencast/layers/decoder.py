"""Decoder layer.

The decoder:
- perform a single message-passing step on mesh2grid using a classical interaction network.
- add a residual connection to the grid nodes.
"""

import torch

from graph_weather.models.gencast.layers.modules import MLP, InteractionNetwork


class Decoder(torch.nn.Module):
    """GenCast's decoder."""

    def __init__(
        self,
        edges_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation_layer: torch.nn.Module = torch.nn.ReLU,
        use_layer_norm: bool = True,
    ):
        """Initialize the Decoder.

        Args:
            edges_dim (int): dimension of edges' features.
            output_dim (int): dimension of final output.
            hidden_dims (list[int]): hidden dimensions of internal MLPs.
            activation_layer (torch.nn.Module, optional): activation function of internal MLPs.
                Defaults to torch.nn.ReLU.
            use_layer_norm (bool, optional): if true add a LayerNorm at the end of each MLP.
                Defaults to True.
        """
        super().__init__()

        # All the MLPs in GenCast have same hidden and output dims. Hence, the embedding latent
        # dimension and the MLPs' output dimension are the same. Moreover, for simplicity, we will
        # ask the hidden dims just once for each MLP in a module: we don't need to specify them
        # individually as arguments, even if the MLPs could have different roles.
        self.latent_dim = hidden_dims[-1]

        # Embedders
        self.edges_mlp = MLP(
            input_dim=edges_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

        # Message Passing
        self.gnn = InteractionNetwork(
            sender_dim=self.latent_dim,
            receiver_dim=self.latent_dim,
            edge_attr_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            use_layer_norm=use_layer_norm,
            activation_layer=activation_layer,
        )

        # Final grid nodes update
        self.grid_mlp_final = MLP(
            input_dim=self.latent_dim,
            hidden_dims=hidden_dims[:-1] + [output_dim],
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

    def forward(
        self,
        input_mesh_nodes: torch.Tensor,
        input_grid_nodes: torch.Tensor,
        input_edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_mesh_nodes (torch.Tensor): mesh nodes' features.
            input_grid_nodes (torch.Tensor): grid nodes' features.
            input_edge_attr (torch.Tensor): grid2mesh edges' features.
            edge_index (torch.Tensor): edge index tensor.

        Returns:
            torch.Tensor: output grid nodes.
        """
        if not (
            input_grid_nodes.shape[-1] == self.latent_dim
            and input_mesh_nodes.shape[-1] == self.latent_dim
        ):
            raise ValueError(
                "The dimension of grid nodes and mesh nodes' features must be "
                "equal to the last hidden dimension."
            )

        # Embedding
        edges_emb = self.edges_mlp(input_edge_attr)

        # Message-passing + residual connection
        latent_grid_nodes = input_grid_nodes + self.gnn(
            x=(input_mesh_nodes, input_grid_nodes),
            edge_index=edge_index,
            edge_attr=edges_emb,
        )

        # Update grid nodes
        latent_grid_nodes = self.grid_mlp_final(latent_grid_nodes)

        return latent_grid_nodes
