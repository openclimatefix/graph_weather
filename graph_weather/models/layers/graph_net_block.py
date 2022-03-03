"""
Functions for building GNN

This code is taken from https://github.com/CCSI-Toolset/MGN which is available under the
US Government License
"""
from typing import Tuple

import torch
from torch import cat, nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum


class MLP(nn.Module):
    """MLP for graph processing"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        MLP

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type one of 'LayerNorm', 'GraphNorm',
                'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP

        Args:
            x: Node or edge features

        Returns:
            The transformed tensor
        """
        return self.model(x)


#############################

# issue with MessagePassing class:
# Only node features are updated after MP iterations
# Need to use MetaLayer to also allow edge features to update


class EdgeProcessor(nn.Module):
    """EdgeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        Edge processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layers
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge, in_dim_edge, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor, u=None, batch=None
    ) -> torch.Tensor:
        """
        Compute the edge part of the message passing

        Args:
            src: Source node tensor
            dest: Destination node tensor
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch Ids, ignored

        Returns:
            The updated edge attributes
        """
        out = cat(
            [src, dest, edge_attr], -1
        )  # concatenate source node, destination node, and edge embeddings
        out = self.edge_mlp(out)
        out += edge_attr  # residual connection

        return out


class NodeProcessor(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u=None, batch=None
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        row, col = edge_index
        out = scatter_sum(edge_attr, col, dim=0)  # aggregate edge message by target
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x  # residual connection

        return out


def build_graph_processor_block(
    in_dim_node: int = 128,
    in_dim_edge: int = 128,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    norm_type: str = "LayerNorm",
) -> torch.nn.Module:
    """
    Build the Graph Net Block

    Args:
        in_dim_node: Input node feature dimension
        in_dim_edge: Input edge feature dimension
        hidden_dim_node: Number of nodes in hidden layer for graph node processing
        hidden_dim_edge: Number of nodes in hidden layer for graph edge processing
        hidden_layers_node: Number of hidden layers for node processing
        hidden_layers_edge: Number of hidden layers for edge processing
        norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    Returns:
        torch.nn.Module for the graph processing block
    """

    return MetaLayer(
        edge_model=EdgeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type
        ),
        node_model=NodeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type
        ),
    )


class GraphProcessor(nn.Module):
    """Overall graph processor"""

    def __init__(
        self,
        mp_iterations: int = 15,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        Graph Processor

        Args:
            mp_iterations: number of message-passing iterations (graph processor blocks)
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim_node: Number of nodes in hidden layers for node processing
            hidden_dim_edge: Number of nodes in hidden layers for edge processing
            hidden_layers_node: Number of hidden layers for node processing
            hidden_layers_edge: Number of hidden layers for edge processing
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(GraphProcessor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type,
                )
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updates to the graph in message passing method

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes

        Returns:
            Updated nodes and edge attributes
        """
        for block in self.blocks:
            x, edge_attr, _ = block(x, edge_index, edge_attr)

        return x, edge_attr
