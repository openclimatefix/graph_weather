"""Dynamic H3 graph construction from latitude and longitude coordinates."""

from typing import List, Optional, Tuple

import h3
import numpy as np
import torch
from torch_geometric.data import Data

from graph_weather.utils import validate_lat_lons


class DynamicGraphBuilder:
    """Build encoder, decoder, and latent graphs from lat/lon coordinates.

    Args:
        resolution: H3 resolution level for the latent mesh.
    """

    def __init__(self, resolution: int = 2):
        """Initialize the builder."""
        self.resolution = resolution
        self.all_h3 = sorted(h3.uncompact_cells(h3.get_res0_cells(), self.resolution))
        self.global_h3_map = {cell: i for i, cell in enumerate(self.all_h3)}
        self._prev_lat_lons: Optional[List[Tuple[float, float]]] = None
        self._cached_encoder_graph: Optional[Data] = None
        self._cached_decoder_graph: Optional[Data] = None
        self._cached_latent_graph: Optional[Data] = None
        self._cached_h3_indices: Optional[List[int]] = None

    def _assign_h3_cells(
        self, lat_lons: List[Tuple[float, float]]
    ) -> Tuple[List[str], List[str], dict]:
        """Map each coordinate to its H3 cell and return index mappings."""
        h3_cells = [h3.latlng_to_cell(lat, lon, self.resolution) for lat, lon in lat_lons]
        unique_cells = sorted(set(h3_cells))
        cell_to_idx = {cell: i for i, cell in enumerate(unique_cells)}
        return h3_cells, unique_cells, cell_to_idx

    def build_encoder_graph(self, lat_lons: List[Tuple[float, float]]) -> Tuple[Data, List[int]]:
        """Build 1-to-1 edges from lat/lon nodes to H3 cells.

        Returns:
            graph: Data with edge_index [2, N] and edge_attr [N, 2].
            h3_indices: Global H3 cell indices for embedding lookup.
        """
        h3_cells, unique_cells, cell_to_idx = self._assign_h3_cells(lat_lons)
        num_coords = len(lat_lons)

        edge_sources = []
        edge_targets = []
        edge_attrs = []

        for node_idx, (coord, cell) in enumerate(zip(lat_lons, h3_cells)):
            edge_sources.append(node_idx)
            edge_targets.append(num_coords + cell_to_idx[cell])
            dist = h3.great_circle_distance(coord, h3.cell_to_latlng(cell), unit="rads")
            edge_attrs.append([np.sin(dist), np.cos(dist)])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        h3_indices = [self.global_h3_map[cell] for cell in unique_cells]

        return Data(edge_index=edge_index, edge_attr=edge_attr), h3_indices

    def build_decoder_graph(self, lat_lons: List[Tuple[float, float]]) -> Data:
        """Build edges from H3 cells to lat/lon nodes.

        Returns:
            Data with edge_index [2, E] and edge_attr [E, 2].
        """
        h3_cells, unique_cells, _ = self._assign_h3_cells(lat_lons)

        all_neighborhood_cells = set()
        for cell in unique_cells:
            all_neighborhood_cells.update(h3.grid_disk(cell, 1))
        all_neighborhood_cells = sorted(all_neighborhood_cells)
        neighborhood_to_idx = {cell: i for i, cell in enumerate(all_neighborhood_cells)}

        edge_sources = []
        edge_targets = []
        edge_attrs = []

        for node_idx, (coord, cell) in enumerate(zip(lat_lons, h3_cells)):
            neighbors = h3.grid_disk(cell, 1)
            for h in neighbors:
                h_idx = neighborhood_to_idx[h]
                edge_sources.append(h_idx)
                edge_targets.append(len(all_neighborhood_cells) + node_idx)
                dist = h3.great_circle_distance(coord, h3.cell_to_latlng(h), unit="rads")
                edge_attrs.append([np.sin(dist), np.cos(dist)])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return Data(edge_index=edge_index, edge_attr=edge_attr)

    def build_latent_graph(self, unique_cells: List[str]) -> Data:
        """Build H3 neighbor edges for the supplied cells.

        Returns:
            Data with edge_index [2, E] and edge_attr [E, 2].
        """
        cell_to_idx = {cell: i for i, cell in enumerate(unique_cells)}

        edge_sources = []
        edge_targets = []
        edge_attrs = []

        for cell in unique_cells:
            neighbors = h3.grid_disk(cell, 1)
            for h in neighbors:
                if h not in cell_to_idx:
                    continue
                dist = h3.great_circle_distance(
                    h3.cell_to_latlng(cell),
                    h3.cell_to_latlng(h),
                    unit="rads",
                )
                edge_sources.append(cell_to_idx[cell])
                edge_targets.append(cell_to_idx[h])
                edge_attrs.append([np.sin(dist), np.cos(dist)])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return Data(edge_index=edge_index, edge_attr=edge_attr)

    def __call__(self, lat_lons: List[Tuple[float, float]]) -> Tuple[Data, Data, Data, List[int]]:
        """Build or return cached encoder, decoder, and latent graphs.

        Returns:
            Tuple of (encoder_graph, decoder_graph, latent_graph, h3_indices).
        """
        if lat_lons is self._prev_lat_lons:
            return (
                self._cached_encoder_graph,
                self._cached_decoder_graph,
                self._cached_latent_graph,
                self._cached_h3_indices,
            )

        validate_lat_lons(lat_lons)

        encoder_graph, h3_indices = self.build_encoder_graph(lat_lons)
        _, unique_cells, _ = self._assign_h3_cells(lat_lons)
        decoder_graph = self.build_decoder_graph(lat_lons)
        latent_graph = self.build_latent_graph(unique_cells)

        self._prev_lat_lons = lat_lons
        self._cached_encoder_graph = encoder_graph
        self._cached_decoder_graph = decoder_graph
        self._cached_latent_graph = latent_graph
        self._cached_h3_indices = h3_indices

        return encoder_graph, decoder_graph, latent_graph, h3_indices
