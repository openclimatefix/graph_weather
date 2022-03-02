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

In further notes, they notice that there is some hexagon instabilities in long rollouts
One possible way to change that is to do the additative noise as in the original MeshGraphNet
or mildly randomize graph connectivity in encoder, as a kind of edge Dropout



"""
import torch
import h3
import numpy as np
from torch_geometric.data import Data, HeteroData

class Encoder(torch.nn.Module):
    def __init__(self, lat_lons: list, resolution: int = 2, input_dim: int = 78, output_dim: int = 256):
        """
        Encode the lat/lon data onto the icosahedron node graph

        Args:
            lat_lons: List of lat/lon pairs
            resolution: Resolution of the h3 grid, int from 0 to 15
            input_dim: Number of input features for the model
            output_dim: Output dimension of the encoded grid
        """
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        self.h3_mapping = {}
        h_index = 0
        for h in self.h3_grid:
            if h not in self.h3_mapping:
                self.h3_mapping[h] = h_index
                h_index += 1
        # Now have the h3 grid mapping, the bipartite graph of edges connecting lat/lon to h3 nodes
        # TODO Add edge features of position of lat/lon nodes to h3 node, which are positions relative to the h3 node
        # Should have vertical and horizontal difference
        self.h3_distances = []
        for idx, h3_point in enumerate(self.h3_grid):
            lat_lon = lat_lons[idx]
            distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit='rads')
            self.h3_distances.append([np.sin(distance), np.cos(distance)])
        self.h3_distances = np.asarray(self.h3_distances)
        # Compress to between 0 and 1

        # Build the default graph
        lat_nodes = torch.zeros((len(lat_lons), input_dim), dtype = torch.float)
        h3_nodes = torch.zeros((h3.num_hexagons(resolution), output_dim), dtype=torch.float)

        # Get connections between lat nodes and h3 nodes
        edge_sources = []
        edge_targets = []
        for node_index, lat_node in enumerate(self.h3_grid):
            edge_sources.append(node_index)
            edge_targets.append(self.h3_mapping[lat_node])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # Use heterogeneous graph as input and output dims are not same for the encoder
        graph = HeteroData()
        graph["latlon"].x = lat_nodes
        graph["iso"].x = h3_nodes
        graph["latlon", "mapped", "iso"].edge_index = edge_index

        graph["latlon", "mapped", "iso"].edge_attr = self.h3_distances
        print(graph)
        self.graph = graph

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

lat_lons = []
for lat in range(-90, 90, 1):
    for lon in range(0, 360, 1):
        lat_lons.append((lat, lon))
print("End create 1 degree grid")
model = Encoder(lat_lons)