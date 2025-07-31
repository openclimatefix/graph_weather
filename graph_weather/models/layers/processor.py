"""Processor for the latent graph

In the original paper the processor is described as

The Processor iteratively processes the 256-channel latent feature data on the icosahedron grid
using 9 rounds of message-passing GNNs. During each round, a node exchanges information with itself
and its immediate neighbors. There are residual connections between each round of processing.

"""

import torch

from graph_weather.models.layers.graph_net_block import GraphProcessor
from graph_weather.models.layers.thermalizer import ThermalizerLayer


class Processor(torch.nn.Module):
    """Processor for latent graphD"""

    def __init__(
        self,
        input_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
        use_thermalizer: bool = False,
    ):
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
            use_thermalizer: Whether to use the thermalizer layer
        """
        super().__init__()
        # Build the default graph
        # Take features from encoder and put into processor graph
        self.input_dim = input_dim
        self.use_thermalizer = use_thermalizer

        self.graph_processor = GraphProcessor(
            num_blocks,
            input_dim,
            edge_dim,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )
        if self.use_thermalizer:
            self.thermalizer = ThermalizerLayer(input_dim)

    def forward(self, x: torch.Tensor, edge_index, edge_attr, t: int = 0) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            x: Torch tensor containing node features
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format
            edge_attr: Edge attribues in [Num edges, Features] shape
            t: Timestep for the thermalizer

        Returns:
            torch Tensor containing the values of the nodes of the graph
        """
        out, _ = self.graph_processor(x, edge_index, edge_attr)
        if self.use_thermalizer:
            out = self.thermalizer(out, t)
        return out
