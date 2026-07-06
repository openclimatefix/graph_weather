"""Stretched-grid regional weather forecaster on a variable-resolution H3 mesh.

This is the single-mesh successor to ``RegionalForecaster``. Instead of one mesh at a single
resolution plus boundary nudging, it runs encode-process-decode over one mesh that is coarse
globally and fine over a chosen region. The coarse/fine seam is stitched by the latent graph,
so surrounding weather flows into the region without an explicit boundary layer. Each cell
starts from a learned embedding held in a global, per-resolution table, so a moving region
reuses the vector it already learned for a location.
"""

from dataclasses import dataclass
from typing import Optional

import h3
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor
from graph_weather.models.layers.processor import Processor
from graph_weather.models.layers.stretched_latent_graph import (
    build_variable_resolution_latent_graph,
)
from graph_weather.models.layers.stretched_mesh import (
    _global_row_map,
    assign_points_to_mesh,
    build_variable_resolution_mesh,
)


@dataclass
class StretchedForecasterConfig:
    """Configuration for StretchedForecaster."""

    coarse_res: int = 2
    fine_res: int = 4
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

    def build(self) -> "StretchedForecaster":
        """Build StretchedForecaster from this configuration."""
        return StretchedForecaster(self)


class StretchedForecaster(nn.Module):
    """Regional forecaster over a variable-resolution ("stretched") H3 mesh."""

    def __init__(self, config: StretchedForecasterConfig):
        """Initialize StretchedForecaster from config."""
        super().__init__()
        self.config = config
        input_dim = config.feature_dim + config.aux_dim
        output_dim = config.output_dim if config.output_dim is not None else config.feature_dim
        self.output_dim = output_dim

        # One learned embedding per H3 cell, kept in a global table per resolution (Option A).
        # forward() gathers the rows for the cells in the current mesh.
        self.coarse_embeddings = nn.Parameter(
            torch.zeros(h3.get_num_cells(config.coarse_res), input_dim)
        )
        self.fine_embeddings = nn.Parameter(
            torch.zeros(h3.get_num_cells(config.fine_res), input_dim)
        )

        # Precompute the global cell->row maps once (mirrors DynamicGraphBuilder), so forward
        # never rebuilds them. Reusing _global_row_map keeps the numbering identical to
        # mixed_resolution_embedding_indices, so a cell's row is stable as the region moves.
        self._coarse_rows = _global_row_map(config.coarse_res)
        self._fine_rows = _global_row_map(config.fine_res)

        # Encoder: obs + cells through a bipartite GNN.
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

        # Processor: message passing cell-to-cell over the latent graph.
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

        # Decoder: cells back to obs through the reversed encoder graph.
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

    def _build_encoder_graph(self, lat_lons: list, mesh: list, mesh_index: dict) -> Data:
        """Build bipartite edges from each observation to its assigned mesh cell."""
        assigned = assign_points_to_mesh(
            lat_lons, mesh, self.config.coarse_res, self.config.fine_res
        )
        num_obs = len(lat_lons)
        sources, targets, attrs = [], [], []
        for node_idx, (coord, cell) in enumerate(zip(lat_lons, assigned)):
            sources.append(node_idx)
            targets.append(num_obs + mesh_index[cell])
            dist = h3.great_circle_distance(coord, h3.cell_to_latlng(cell), unit="rads")
            attrs.append([np.sin(dist), np.cos(dist)])
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
        return Data(edge_index=edge_index, edge_attr=edge_attr)

    def _gather_embeddings(self, mesh: list, device: torch.device) -> torch.Tensor:
        """Gather each mesh cell's embedding from the coarse or fine global table.

        Cells are looked up in the row maps precomputed at init, so no global cell numbering
        is rebuilt per call.
        """
        coarse_pos, coarse_rows, fine_pos, fine_rows = [], [], [], []
        for pos, cell in enumerate(mesh):
            if h3.get_resolution(cell) == self.config.fine_res:
                fine_pos.append(pos)
                fine_rows.append(self._fine_rows[cell])
            else:
                coarse_pos.append(pos)
                coarse_rows.append(self._coarse_rows[cell])

        embeds = torch.zeros(len(mesh), self.coarse_embeddings.shape[1], device=device)
        if coarse_pos:
            embeds[coarse_pos] = self.coarse_embeddings[coarse_rows]
        if fine_pos:
            embeds[fine_pos] = self.fine_embeddings[fine_rows]
        return embeds

    def forward(
        self,
        features: torch.Tensor,
        lat_lons: list,
        bbox: tuple,
    ) -> torch.Tensor:
        """Forecast the next state over a stretched mesh refined on ``bbox``.

        Args:
            features: Input features [B, N_obs, feature_dim + aux_dim].
            lat_lons: List of (lat, lon) for the observations.
            bbox: Region to refine as (lat_min, lat_max, lon_min, lon_max) in degrees.

        Returns:
            Predicted next state [B, N_obs, output_dim].
        """
        batch_size = features.shape[0]
        num_obs = features.shape[1]
        device = features.device

        mesh = build_variable_resolution_mesh(bbox, self.config.coarse_res, self.config.fine_res)
        mesh_index = {cell: i for i, cell in enumerate(mesh)}

        enc_graph = self._build_encoder_graph(lat_lons, mesh, mesh_index).to(device)
        lat_graph = build_variable_resolution_latent_graph(mesh).to(device)
        mesh_embeds = self._gather_embeddings(mesh, device)

        enc_edge_attr = self.edge_encoder(enc_graph.edge_attr)
        latent_edge_attr = self.latent_edge_encoder(lat_graph.edge_attr)

        # Decoder reuses the encoder edges reversed: same nodes, opposite direction.
        dec_edge_index = enc_graph.edge_index.flip(0)
        dec_edge_attr = self.decoder_edge_encoder(enc_graph.edge_attr)

        batch_outputs = []
        for i in range(batch_size):
            nodes = torch.cat([features[i], mesh_embeds], dim=0)
            nodes = self.node_encoder(nodes)
            nodes, _ = self.encoder_gnn(nodes, enc_graph.edge_index, enc_edge_attr)
            cell_features = nodes[num_obs:]

            cell_features = self.processor(cell_features, lat_graph.edge_index, latent_edge_attr)

            obs_placeholders = torch.zeros(num_obs, self.config.node_dim, device=device)
            dec_nodes = torch.cat([obs_placeholders, cell_features], dim=0)
            dec_nodes, _ = self.decoder_gnn(dec_nodes, dec_edge_index, dec_edge_attr)
            obs_out = self.node_decoder(dec_nodes[:num_obs])
            batch_outputs.append(obs_out)

        out = torch.stack(batch_outputs, dim=0)

        # Residual: predict the change and add it to the input.
        out = out + features[..., : self.output_dim]
        return out
