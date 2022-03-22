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
import einops
import h3
import numpy as np
import torch
from torch_geometric.data import Data

from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor


class AssimilatorDecoder(torch.nn.Module):
    """Assimilator graph module"""

    def __init__(
        self,
        lat_lons,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        output_edge_dim: int = 256,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
    ):
        """
        Decoder from latent graph to lat/lon graph for assimilation of observation

        Args:
            lat_lons: List of (lat,lon) points
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        self.num_latlons = len(lat_lons)
        self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
        self.num_h3 = len(self.base_h3_grid)
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        self.h3_to_index = {}
        h_index = len(self.base_h3_grid)
        for h in self.base_h3_grid:
            if h not in self.h3_to_index:
                h_index -= 1
                self.h3_to_index[h] = h_index
        self.h3_mapping = {}
        for h, value in enumerate(self.h3_grid):
            self.h3_mapping[h + self.num_h3] = value

        # Build the default graph
        nodes = torch.zeros(
            (len(lat_lons) + h3.num_hexagons(resolution), input_dim), dtype=torch.float
        )
        # Extra starting ones for appending to inputs, could 'learn' good starting points
        self.latlon_nodes = torch.zeros((len(lat_lons), input_dim), dtype=torch.float)
        # Get connections between lat nodes and h3 nodes TODO Paper makes it seem like the 3
        #  closest iso points map to the lat/lon point Do kring 1 around current h3 cell,
        #  and calculate distance between all those points and the lat/lon one, choosing the
        #  nearest N (3) For a bit simpler, just include them all with their distances
        edge_sources = []
        edge_targets = []
        self.h3_to_lat_distances = []
        for node_index, h_node in enumerate(self.h3_grid):
            # Get h3 index
            h_points = h3.k_ring(self.h3_mapping[node_index + self.num_h3], 1)
            for h in h_points:
                distance = h3.point_dist(lat_lons[node_index], h3.h3_to_geo(h), unit="rads")
                self.h3_to_lat_distances.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(self.h3_to_index[h])
                edge_targets.append(node_index + self.num_h3)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.h3_to_lat_distances = torch.tensor(self.h3_to_lat_distances, dtype=torch.float)

        # Use normal graph as its a bit simpler
        self.graph = Data(x=nodes, edge_index=edge_index, edge_attr=self.h3_to_lat_distances)

        self.edge_encoder = MLP(2, output_edge_dim, hidden_dim_processor_edge, 2, mlp_norm_type)
        self.graph_processor = GraphProcessor(
            mp_iterations=1,
            in_dim_node=input_dim,
            in_dim_edge=output_edge_dim,
            hidden_dim_node=hidden_dim_processor_node,
            hidden_dim_edge=hidden_dim_processor_edge,
            hidden_layers_node=hidden_layers_processor_node,
            hidden_layers_edge=hidden_layers_processor_edge,
            norm_type=mlp_norm_type,
        )
        self.node_decoder = MLP(
            input_dim, output_dim, hidden_dim_decoder, hidden_layers_decoder, None
        )

    def forward(self, processor_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            processor_features: Processed features in shape [B*Nodes, Features]

        Returns:
            Updated features for model
        """
        edge_attr = self.edge_encoder(self.graph.edge_attr)  # Update attributes based on distance
        edge_attr = einops.repeat(edge_attr, "e f -> (repeat e) f", repeat=batch_size)

        edge_index = torch.cat(
            [
                self.graph.edge_index + i * torch.max(self.graph.edge_index) + i
                for i in range(batch_size)
            ],
            dim=1,
        )

        # Readd nodes to match graph node number
        self.latlon_nodes = self.latlon_nodes.to(processor_features.device)
        features = einops.rearrange(processor_features, "(b n) f -> b n f", b=batch_size)
        features = torch.cat(
            [features, einops.repeat(self.latlon_nodes, "n f -> b n f", b=batch_size)], dim=1
        )
        features = einops.rearrange(features, "b n f -> (b n) f")

        out, _ = self.graph_processor(features, edge_index, edge_attr)  # Message Passing
        # Remove the h3 nodes now, only want the latlon ones
        out = self.node_decoder(out)  # Decode to 78 from 256
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        test, out = torch.split(out, [self.num_h3, self.num_latlons], dim=1)
        return out
