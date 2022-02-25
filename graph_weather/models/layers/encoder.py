"""Encoders to encode from the input graph to the latent graph

In the original paper the encoder is described as

The Encoder maps from physical data defined on a latitude/longitude grid to abstract latent features
defined on an icosahedron grid.  The Encoder GNN uses a bipartite graph(lat/lon→icosahedron) with
edges only between nodes in the lat/lon grid and nodes in the icosahedron grid. Put another way,
spatial and channel information in the local neighborhood of each icosahedron node is
gathered using connections to nearby lat/lon nodes.

The initial node features are the 78 atmospheric variables described in Section 2.1, plus solar
radiation, orography, land-sea mask, the day-of-year,sin(lat),cos(lat),sin(lon), and cos(lon).
The initial edge features are the positions of the lat/lon nodes connected to each icosahedron node.
These positions are provided in a local coordinate system that is defined relative
to each icosahedron node.

"""
import torch
import h3
from torch_geometric.data import Data

class Encoder(torch.nn.Module):
    def __init__(self, lat_lons: list, resolution: int = 2, output_dim: int = 256):
        """
        Encode the lat/lon data onto the icosahedron node graph

        Args:
            lat_lons: List of lat/lon pairs
            resolution: Resolution of the h3 grid, int from 0 to 15
            output_dim: Output dimension of the encoded grid
        """
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        # Now have the h3 grid mapping, the bipartite graph of edges connecting lat/lon to h3 nodes
        # TODO Add edge features of position of lat/lon nodes to h3 node, which are positions relative to the h3 node

        # TODO Add MLP to convert to 256 dim output
        super().__init__()

    def forward(self, features):
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:

        """

        # TODO Add node features based on the variables desired

        return NotImplementedError