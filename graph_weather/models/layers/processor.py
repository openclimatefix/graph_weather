"""Processor for the latent graphy

In the original paper the processor is described as

The Processor iteratively processes the 256-channel latent feature data on the icosahedron grid using
9 rounds of message-passing GNNs. During each round, a node exchanges information with itself
and its immediate neighbors. There are residual connections between each round of processing.

"""
import torch

from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor


class Processor(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        edge_dim: int = 1,
        num_blocks: int = 9,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
    ):
        """
        Process the latent graph multiple times

        Args:

        """
        super().__init__()
        # Build the default graph
        # Take features from encoder and put into processor graph
        self.input_dim = input_dim

        # TODO Add MLP to convert to 256 dim processor input to the original feature output
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

    def forward(self, x: torch.Tensor, edge_index, edge_attr) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            x: Torch tensor containing node features
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format
            edge_attr: Edge attribues in [Num edges, Features] shape

        Returns:
            torch Tensor containing the values of the nodes of the graph
        """
        out, _ = self.graph_processor(x, edge_index, edge_attr)
        return out
