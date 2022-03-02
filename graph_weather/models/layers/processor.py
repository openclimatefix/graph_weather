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
        input_dim: int = 256,
        output_dim: int = 256,
        num_blocks: int = 9,
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

        # TODO Add MLP to convert to 256 dim processor input to the original feature output
        super().__init__()

    def forward(self, graph: Data):
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:

        """
        # TODO Process with message passing blocks
        return NotImplementedError
