"""
Processor for the latent graph. In the original paper the processor is described as:

The Processor iteratively processes the 256-channel latent feature data on the icosahedron grid
using 9 rounds of message-passing GNNs. During each round, a node exchanges information with itself
and its immediate neighbors. There are residual connections between each round of processing.
"""
from typing import Optional

import torch

from graph_weather.models.layers.gnn_blocks import GraphProcessor


class Processor(torch.nn.Module):
    """Processor for latent graph."""

    def __init__(
        self,
        input_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: Optional[str] = "LayerNorm",
    ) -> None:
        """
        Latent graph processor

        Args:
            input_dim: Input dimension for the node
            edge_dim: Edge input dimension
            num_blocks: Number of message passing blocks
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()

        # Build the default graph
        # Take features from encoder and put into processor graph
        self.input_dim = input_dim

        self.graph_processor = GraphProcessor(
            mp_iterations=num_blocks,
            in_dim_node=input_dim,
            in_dim_edge=edge_dim,
            hidden_dim_node=hidden_dim_processor_node,
            hidden_dim_edge=hidden_dim_processor_edge,
            hidden_layers_node=hidden_layers_processor_node,
            hidden_layers_edge=hidden_layers_processor_edge,
            norm_type=mlp_norm_type,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            x: tensor containing node features
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format
            edge_attr: Edge attributes in [Num edges, Features] shape

        Returns:
            Values of the nodes of the graph.
        """
        out, _ = self.graph_processor(x, edge_index, edge_attr)
        return out
