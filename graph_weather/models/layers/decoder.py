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
import h3
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

from graph_weather.models.layers.encoder import Encoder


class Decoder(torch.nn.Module):
    def __init__(self, lat_lons, resolution: int = 2, input_dim: int = 256, output_dim: int = 78):
        """
        Decode the lat/lon data onto the icosahedron node graph

        Args:
            h3_to_latlon: Bipartite mapping from h3 indicies to lat/lon in original grid
            processor_dim: Processor output dim size
            feature_dim: Output dimension of the original graph
        """
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        self.index_to_h3 = {}
        h_index = 0
        for h in self.h3_grid:
            if h not in self.index_to_h3:
                self.index_to_h3[h] = h_index
                h_index += 1
        self.h3_mapping = {}
        for h, value in enumerate(self.h3_grid):
            self.h3_mapping[h] = value

        # Build the default graph
        lat_nodes = torch.zeros((len(lat_lons), output_dim), dtype=torch.float)
        h3_nodes = torch.zeros((h3.num_hexagons(resolution), input_dim), dtype=torch.float)

        # Get connections between lat nodes and h3 nodes
        # TODO Paper makes it seem like the 3 closest iso points map to teh lat/lon point
        # Do kring 1 around current h3 cell, and calculate distance between all those points and the lat/lon one, choosing the nearest N (3)
        # For a bit simpler, just include them all with their distances
        edge_sources = []
        edge_targets = []
        self.h3_to_lat_distances = []
        for node_index, lat_node in enumerate(lat_lons):
            # Get h3 index
            h_points = h3.k_ring(self.h3_mapping[node_index], 1)
            for h in h_points:
                distance = h3.point_dist(lat_node, h3.h3_to_geo(h), unit="rads")
                self.h3_to_lat_distances.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(self.index_to_h3[h])
                edge_targets.append(node_index)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.h3_to_lat_distances = np.asarray(self.h3_to_lat_distances)

        # Use heterogeneous graph as input and output dims are not same for the encoder
        graph = HeteroData()
        graph["latlon"].x = lat_nodes
        graph["iso"].x = h3_nodes
        graph["iso", "mapped", "latlon"].edge_index = edge_index

        graph["iso", "mapped", "latlon"].edge_attr = self.h3_to_lat_distances
        print(graph)
        self.graph = graph

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
