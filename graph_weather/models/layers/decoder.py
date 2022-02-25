"""Decoders to decode from the Processor graph to the original graph with updated values

In the original paper the decoder is described as

The Decoder maps back to physical data defined on a latitude/longitude grid. The underlying graph is
again bipartite, this time mapping icosahedron→lat/lon.
The inputs to the Decoder come from the Processor, plus a skip connection back to the original
state of the 78 atmospheric variables onthe latitude/longitude grid.
The output of the Decoder is the predicted 6-hour change in the 78 atmospheric variables,
which is then added to the initial state to produce the new state. We found 6 hours to be a good
balance between shorter time steps (simpler dynamics to model but more iterations required during
rollout) and longer time steps (fewer iterations required during rollout but modeling
more complex dynamics)

"""
import torch
import h3
from torch_geometric.data import Data

class Decoder(torch.nn.Module):
    def __init__(self, h3_to_latlon, processor_dim, feature_dim):
        """
        Decode the lat/lon data onto the icosahedron node graph

        Args:
            h3_to_latlon: Bipartite mapping from h3 indicies to lat/lon in original grid
            processor_dim: Processor output dim size
            feature_dim: Output dimension of the original graph
        """
        self.h3_to_latlon = h3_to_latlon
        # TODO Add MLP to convert to 256 dim processor input to the original feature output
        super().__init__()

    def forward(self, original_graph, processor_graph):
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:

        """

        # TODO Add node features based on the variables desired
        out = processor_graph
        # TODO Have skip connection to original graph
        out += original_graph
        return NotImplementedError