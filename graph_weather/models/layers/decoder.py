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
from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor



class Decoder(torch.nn.Module):
    def __init__(self, lat_lons, resolution: int = 2, input_dim: int = 256, output_dim: int = 78,
                 hidden_dim_processor_node=256,
                 hidden_dim_processor_edge=256,
                 hidden_layers_processor_node=2,
                 hidden_layers_processor_edge=2,
                 mlp_norm_type="LayerNorm",
                 hidden_dim_decoder=128,
                 hidden_layers_decoder=2,):
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
                self.index_to_h3[h] = h_index + len(lat_lons)
                h_index += 1
        self.h3_mapping = {}
        for h, value in enumerate(self.h3_grid):
            self.h3_mapping[h] = value

        # Build the default graph
        nodes = torch.zeros((len(lat_lons) + h3.num_hexagons(resolution), input_dim), dtype=torch.float)

        # Get connections between lat nodes and h3 nodes
        # TODO Paper makes it seem like the 3 closest iso points map to the lat/lon point
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

        # Use normal graph as its a bit simpler
        self.graph = Data(x=nodes, edge_index = edge_index, edge_attr = self.h3_to_lat_distances)

        self.edge_encoder = MLP(
            2, 2, 256, 2, mlp_norm_type
            )
        self.graph_processor = GraphProcessor(
            1,
            output_dim,
            2,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
            )
        self.node_decoder = MLP(
            input_dim, output_dim, hidden_dim_decoder, hidden_layers_decoder, None
            )
        super().__init__()

    def forward(self, processor_features: torch.Tensor, start_features: torch.Tensor) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            processor_features: Processed features
            start_features: Original input features to the encoder

        Returns:
            Updated features for model
        """

        edge_attr = self.edge_encoder(self.graph.edge_attr) # Update attributes based on distance
        out, _ = self.graph_processor(processor_features, self.graph.edge_index, edge_attr) # Message Passing
        out = self.node_decoder(out) # Decode to 78 from 256
        out += start_features
        return out
