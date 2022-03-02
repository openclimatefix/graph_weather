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
import einops
import h3
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor


class Encoder(torch.nn.Module):
    def __init__(
        self,
        lat_lons: list,
        resolution: int = 2,
        input_dim: int = 78,
        output_dim: int = 256,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
    ):
        """
        Encode the lat/lon data onto the icosahedron node graph

        Args:
            lat_lons: List of lat/lon pairs
            resolution: Resolution of the h3 grid, int from 0 to 15
            input_dim: Number of input features for the model
            output_dim: Output dimension of the encoded grid
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_latlons = len(lat_lons)
        self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        self.h3_mapping = {}
        h_index = 0
        for h in self.base_h3_grid:
            if h not in self.h3_mapping:
                self.h3_mapping[h] = h_index + self.num_latlons
                h_index += 1
        # Now have the h3 grid mapping, the bipartite graph of edges connecting lat/lon to h3 nodes
        # TODO Add edge features of position of lat/lon nodes to h3 node, which are positions relative to the h3 node
        # Should have vertical and horizontal difference
        self.h3_distances = []
        for idx, h3_point in enumerate(self.h3_grid):
            lat_lon = lat_lons[idx]
            distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit="rads")
            self.h3_distances.append([np.sin(distance), np.cos(distance)])
        self.h3_distances = torch.tensor(self.h3_distances, dtype=torch.float)
        # Compress to between 0 and 1

        # Build the default graph
        # lat_nodes = torch.zeros((len(lat_lons), input_dim), dtype=torch.float)
        # h3_nodes = torch.zeros((h3.num_hexagons(resolution), output_dim), dtype=torch.float)
        nodes = torch.zeros(
            (len(lat_lons) + h3.num_hexagons(resolution), input_dim), dtype=torch.float
        )
        # Get connections between lat nodes and h3 nodes
        edge_sources = []
        edge_targets = []
        for node_index, lat_node in enumerate(self.h3_grid):
            edge_sources.append(node_index)
            edge_targets.append(self.h3_mapping[lat_node])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # Use homogenous graph to make it easier
        self.graph = Data(x=nodes, edge_index=edge_index, edge_attr=self.h3_distances)

        self.latent_graph = self.create_latent_graph()

        # Extra starting ones for appending to inputs, could 'learn' good starting points
        self.h3_nodes = torch.zeros((h3.num_hexagons(resolution), output_dim), dtype=torch.float)
        # Output graph

        self.node_encoder = MLP(input_dim, output_dim, 256, 2, mlp_norm_type)
        self.edge_encoder = MLP(2, 2, 256, 2, mlp_norm_type)
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

    def forward(self, features: torch.Tensor):
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:

        """
        out = self.node_encoder(features)  # Encode to 256 from 78
        edge_attr = self.edge_encoder(self.graph.edge_attr)  # Update attributes based on distance
        # Cat with the h3 nodes to have correct amount of nodes, and in right order
        out = torch.cat([out, self.h3_nodes], dim=0)
        out, _ = self.graph_processor(out, self.graph.edge_index, edge_attr)  # Message Passing
        # Remove the extra nodes (lat/lon) from the output
        _, out = torch.split(out, [self.num_latlons, self.h3_nodes.shape[0]], dim=0)
        return out, self.latent_graph.edge_index, self.latent_graph.edge_attr  # New graph

    def create_latent_graph(self) -> Data:
        """
        Copies over and generates a Data object for the processor to use
        Args:
            graph:

        Returns:

        """
        # Get connectivity of the graph
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for h3_index in self.h3_mapping.keys():
            h_points = h3.k_ring(h3_index, 1)
            for h in h_points:  # Already includes itself
                distance = h3.point_dist(h3.h3_to_geo(h3_index), h3.h3_to_geo(h), unit="rads")
                edge_attrs.append(distance)
                edge_sources.append(self.h3_mapping[h3_index] - self.num_latlons)
                edge_targets.append(self.h3_mapping[h] - self.num_latlons)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.unsqueeze(torch.tensor(edge_attrs, dtype=torch.float), dim=-1)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        return Data(edge_index=edge_index, edge_attr=edge_attrs)
