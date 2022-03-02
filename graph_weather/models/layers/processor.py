"""Processor for the latent graphy

In the original paper the processor is described as

The Processor iteratively processes the 256-channel latent feature data on the icosahedron grid using
9 rounds of message-passing GNNs. During each round, a node exchanges information with itself
and its immediate neighbors. There are residual connections between each round of processing.

"""
import h3
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData


class Processor(torch.nn.Module):
    def __init__(
        self,
        h3_mapping: dict,
        input_dim: int = 256,
        output_dim: int = 256,
        num_blocks: int = 9,
        num_neighbors: int = 1,
    ):
        """
        Process the latent graph multiple times before sending it to the decoder

        Args:
            h3_to_latlon: Bipartite mapping from h3 indicies to lat/lon in original grid
            processor_dim: Processor output dim size
            feature_dim: Output dimension of the original graph
        """

        # Build the default graph
        # Take features from encoder and put into processor graph
        self.input_dim = input_dim

        # Get connectivity of the graph
        edge_sources = []
        edge_targets = []
        for h3_index in h3_mapping.keys():
            h_points = h3.k_ring(h3_index, 1)
            for h in h_points:  # Already includes itself
                edge_sources.append(h3_mapping[h3_index])
                edge_targets.append(h3_mapping[h])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        h3_nodes = torch.zeros((len(h3_mapping), input_dim), dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        self.graph = Data(x=h3_nodes, edge_index=edge_index)
        print(self.graph)

        # TODO Add MLP to convert to 256 dim processor input to the original feature output
        super().__init__()

    def forward(self, graph: HeteroData, h3_mapping: dict):
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:

        """
        # TODO copy over the iso grid features to the processing grid
        edge_sources = []
        edge_targets = []
        for h3_index in h3_mapping.keys():
            h_points = h3.k_ring(h3_index, 1)
            for h in h_points:  # Already includes itself
                edge_sources.append(h3_mapping[h3_index])
                edge_targets.append(h3_mapping[h])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        self.graph = Data(x=graph["iso"].x, edge_index=edge_index)
        # TODO Process with message passing blocks
        return NotImplementedError
