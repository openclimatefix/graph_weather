"""Encoder layer.

The encoder:
- embeds grid nodes, mesh nodes and g2m edges' features to the latent space.
- perform a single message-passing step using a classical interaction network.
- add a residual connection to the mesh and grid nodes.
"""

import torch

from graph_weather.models.gencast.layers.modules import MLP, InteractionNetwork


class Encoder(torch.nn.Module):
    """GenCast's encoder."""

    def __init__(
        self,
        grid_dim: int,
        mesh_dim: int,
        edge_dim: int,
        hidden_dims: list[int],
        activation_layer: torch.nn.Module = torch.nn.ReLU,
        use_layer_norm: bool = True,
    ):
        """Initialize the Encoder.

        Args:
            grid_dim (int): dimension of grid nodes' features.
            mesh_dim (int): dimension of mesh nodes' features
            edge_dim (int): dimension of g2m edges' features
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
        self.grid_mlp = MLP(
            input_dim=grid_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

        self.mesh_mlp = MLP(
            input_dim=mesh_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

        self.edges_mlp = MLP(
            input_dim=edge_dim,
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
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

    def forward(
        self,
        input_grid_nodes: torch.Tensor,
        input_mesh_nodes: torch.Tensor,
        input_edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_grid_nodes (torch.Tensor): grid nodes' features.
            input_mesh_nodes (torch.Tensor): mesh nodes' features.
            input_edge_attr (torch.Tensor): grid2mesh edges' features.
            edge_index (torch.Tensor): edge index tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output grid nodes, output mesh nodes.
        """

        # Embedding
        grid_emb = self.grid_mlp(input_grid_nodes)
        mesh_emb = self.mesh_mlp(input_mesh_nodes)
        edges_emb = self.edges_mlp(input_edge_attr)

        # Message-passing + residual connection
        latent_mesh_nodes = mesh_emb + self.gnn(
            x=(grid_emb, mesh_emb),
            edge_index=edge_index,
            edge_attr=edges_emb,
        )

        # Update grid nodes + residual connection
        latent_grid_nodes = grid_emb + self.grid_mlp_final(grid_emb)

        return latent_grid_nodes, latent_mesh_nodes
