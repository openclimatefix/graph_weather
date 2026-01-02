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

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        edge_attr,
        t: int = 0,
        batch_size: int = None,
        efficient_batching: bool = False,
    ) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            x: Torch tensor containing node features [B*N, F] or [N, F]
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format
            edge_attr: Edge attributes in [Num edges, Features] shape
            t: Timestep for the thermalizer
            batch_size: Batch size (required when efficient_batching=True)
            efficient_batching: If True, process batches separately with shared graph

        Returns:
            torch Tensor containing the values of the nodes of the graph
        """
        if efficient_batching and batch_size is not None and batch_size > 1:
            # Efficient batching: process each batch separately with shared graph
            # x is [B*N, F], split into B batches of [N, F]
            num_nodes_per_batch = x.shape[0] // batch_size
            x_batched = x.view(batch_size, num_nodes_per_batch, -1)

            batch_outputs = []
            for i in range(batch_size):
                # Process single batch with shared graph
                out_i, _ = self.graph_processor(x_batched[i], edge_index, edge_attr)
                if self.use_thermalizer:
                    out_i = self.thermalizer(out_i, t)
                batch_outputs.append(out_i)

            # Concatenate outputs back to [B*N, F] format
            out = torch.cat(batch_outputs, dim=0)
            return out
        else:
            # Original batching: process all at once with batched graph
            out, _ = self.graph_processor(x, edge_index, edge_attr)
            if self.use_thermalizer:
                out = self.thermalizer(out, t)
            return out
