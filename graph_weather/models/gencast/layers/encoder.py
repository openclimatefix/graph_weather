"""Encoder layer.

The encoder:
- embeds grid nodes, mesh nodes and g2m edges' features to the latent space.
- perform a single message-passing step using a classical interaction network.
- add a residual connection to the mesh nodes.
"""
import torch
from torch.nn.modules import 

class Encoder(torch.nn.Module):
    def __init__(self, 
                 grid_nodes_input_dim, 
                 mesh_nodes_input_dim,
                 edge_attr_input_dim,
                 mlp_hidden_dim,
                 latent_dim,
                 mlp_norm_type="LayerNorm",
                 mlp_act_function="swish"):
        super().__init__()

        # Embedders
        self.grid_nodes_mlp = MLP([grid_nodes_input_dim, mlp_hidden_dim, latent_dim],
                                  norm = mlp_norm_type,
                                  act=mlp_act_function)
        self.mesh_nodes_mlp = MLP([mesh_nodes_input_dim, mlp_hidden_dim, latent_dim],
                                  norm = mlp_norm_type,
                                  act=mlp_act_function)
        self.edge_attr_mlp = MLP([edge_attr_input_dim, mlp_hidden_dim, latent_dim],
                                  norm = mlp_norm_type, 
                                  act=mlp_act_function)
        
        # Message Passing
        self.conv = InteractionNetwork(senders_input_dim=latent_dim,
                                    receivers_input_dim=latent_dim,
                                    edges_input_dim=latent_dim, 
                                    hidden_dim=mlp_hidden_dim, 
                                    output_dim=latent_dim,
                                    mlp_norm_type=mlp_norm_type,
                                    mlp_act_function=mlp_act_function,
                                    )
        
        self.grid_nodes_mlp_2 = MLP([latent_dim, mlp_hidden_dim, latent_dim],
                                    norm=mlp_norm_type,
                                    act=mlp_act_function)
       
    def forward(self, input_grid_nodes, input_mesh_nodes, input_edge_attr, edge_index):
        # Embedding
        grid_nodes_emb = self.grid_nodes_mlp(input_grid_nodes)
        mesh_nodes_emb = self.mesh_nodes_mlp(input_mesh_nodes)
        edge_attr_emb = self.edge_attr_mlp(input_edge_attr)
    
        latent_mesh_nodes = mesh_nodes_emb + self.conv(x=(grid_nodes_emb, mesh_nodes_emb),
                                 edge_index=edge_index,
                                 edge_attr=edge_attr_emb)
        latent_grid_nodes = grid_nodes_emb + self.grid_nodes_mlp_2(grid_nodes_emb)
        # TODO: Ask why we need to update eg2m, since we don't use them later
        return latent_mesh_nodes, latent_grid_nodes
