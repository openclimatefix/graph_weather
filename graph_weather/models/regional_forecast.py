"""Regional weather forecaster with movable high-res domain."""

from dataclasses import dataclass
from typing import Optional

import h3
import torch
import torch.nn as nn

from graph_weather.models.layers.dynamic_graph_builder import DynamicGraphBuilder
from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor
from graph_weather.models.layers.processor import Processor


@dataclass
class RegionalForecasterConfig:
    """Configuration for RegionalForecaster."""

    resolution: int = 2
    feature_dim: int = 78
    aux_dim: int = 24
    output_dim: Optional[int] = None
    node_dim: int = 256
    edge_dim: int = 256
    num_blocks: int = 9
    hidden_dim_processor_node: int = 256
    hidden_dim_processor_edge: int = 256
    hidden_layers_processor_node: int = 2
    hidden_layers_processor_edge: int = 2
    hidden_dim_decoder: int = 128
    hidden_layers_decoder: int = 2
    norm_type: str = "LayerNorm"
    use_checkpointing: bool = False

    def build(self) -> "RegionalForecaster":
        """Build RegionalForecaster from this configuration."""
        return RegionalForecaster(self)


class RegionalForecaster(nn.Module):
    """Regional weather forecaster with dynamic graph construction."""

    def __init__(self, config: RegionalForecasterConfig):
        """Initialize RegionalForecaster from config."""
        super().__init__()
        self.config = config
        input_dim = config.feature_dim + config.aux_dim
        output_dim = config.output_dim if config.output_dim is not None else config.feature_dim
        self.output_dim = output_dim

        self.graph_builder = DynamicGraphBuilder(resolution=config.resolution)

        # Learnable embedding per H3 cell; forward() indexes the regional subset
        self.h3_embeddings = nn.Parameter(
            torch.zeros(h3.get_num_cells(config.resolution), input_dim)
        )

        # Encoder
        self.node_encoder = MLP(
            input_dim,
            config.node_dim,
            config.hidden_dim_processor_node,
            config.hidden_layers_processor_node,
            config.norm_type,
            config.use_checkpointing,
        )
        self.edge_encoder = MLP(
            2,
            config.edge_dim,
            config.hidden_dim_processor_edge,
            config.hidden_layers_processor_edge,
            config.norm_type,
            config.use_checkpointing,
        )
        self.encoder_gnn = GraphProcessor(
            1,
            config.node_dim,
            config.edge_dim,
            config.hidden_dim_processor_node,
            config.hidden_dim_processor_edge,
            config.hidden_layers_processor_node,
            config.hidden_layers_processor_edge,
            config.norm_type,
            use_checkpointing=config.use_checkpointing,
        )

        # Processor
        self.latent_edge_encoder = MLP(
            2,
            config.edge_dim,
            config.hidden_dim_processor_edge,
            config.hidden_layers_processor_edge,
            config.norm_type,
            config.use_checkpointing,
        )
        self.processor = Processor(
            input_dim=config.node_dim,
            edge_dim=config.edge_dim,
            num_blocks=config.num_blocks,
            hidden_dim_processor_edge=config.hidden_dim_processor_edge,
            hidden_layers_processor_node=config.hidden_layers_processor_node,
            hidden_dim_processor_node=config.hidden_dim_processor_node,
            hidden_layers_processor_edge=config.hidden_layers_processor_edge,
            mlp_norm_type=config.norm_type,
        )

        # Decoder
        self.decoder_edge_encoder = MLP(
            2,
            config.edge_dim,
            config.hidden_dim_processor_edge,
            config.hidden_layers_processor_edge,
            config.norm_type,
            config.use_checkpointing,
        )
        self.decoder_gnn = GraphProcessor(
            1,
            config.node_dim,
            config.edge_dim,
            config.hidden_dim_processor_node,
            config.hidden_dim_processor_edge,
            config.hidden_layers_processor_node,
            config.hidden_layers_processor_edge,
            config.norm_type,
            use_checkpointing=config.use_checkpointing,
        )
        self.node_decoder = MLP(
            config.node_dim,
            output_dim,
            config.hidden_dim_decoder,
            config.hidden_layers_decoder,
            config.norm_type,
            config.use_checkpointing,
        )

    def forward(
        self,
        features: torch.Tensor,
        lat_lons: list,
    ) -> torch.Tensor:
        """
        Regional weather forecast for a given set of coordinates.

        Args:
            features: Input features [B, N_obs, feature_dim + aux_dim]
            lat_lons: List of (lat, lon) for this region

        Returns:
            Predicted next state [B, N_obs, output_dim]
        """
        batch_size = features.shape[0]
        num_obs = features.shape[1]

        # Build dynamic graphs from coordinates
        enc_graph, _, lat_graph, h3_indices = self.graph_builder(lat_lons)
        enc_graph = enc_graph.to(features.device)
        lat_graph = lat_graph.to(features.device)

        # Get regional H3 embeddings from global table
        regional_h3 = self.h3_embeddings[h3_indices]

        # Encode edges (shared across batch)
        enc_edge_attr = self.edge_encoder(enc_graph.edge_attr)
        latent_edge_attr = self.latent_edge_encoder(lat_graph.edge_attr)

        # Decoder uses reversed encoder edges: same nodes, opposite direction
        dec_edge_index = enc_graph.edge_index.flip(0)
        dec_edge_attr = self.decoder_edge_encoder(enc_graph.edge_attr)

        batch_outputs = []
        for i in range(batch_size):
            # Encode: obs + H3 nodes through bipartite GNN
            nodes = torch.cat([features[i], regional_h3], dim=0)
            nodes = self.node_encoder(nodes)
            nodes, _ = self.encoder_gnn(nodes, enc_graph.edge_index, enc_edge_attr)
            h3_features = nodes[num_obs:]

            # Process: N rounds of H3 message passing
            h3_features = self.processor(h3_features, lat_graph.edge_index, latent_edge_attr)

            # Decode: H3 -> obs through reversed bipartite GNN
            obs_placeholders = torch.zeros(num_obs, self.config.node_dim, device=features.device)
            dec_nodes = torch.cat([obs_placeholders, h3_features], dim=0)
            dec_nodes, _ = self.decoder_gnn(dec_nodes, dec_edge_index, dec_edge_attr)
            obs_out = self.node_decoder(dec_nodes[:num_obs])
            batch_outputs.append(obs_out)

        out = torch.stack(batch_outputs, dim=0)

        # Residual: predict delta, add to input
        out = out + features[..., : self.output_dim]

        return out
