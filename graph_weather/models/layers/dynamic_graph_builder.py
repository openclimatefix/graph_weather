"""Dynamic bipartite graph construction for regional weather models.

Builds lat/lon ↔ H3 bipartite graphs at call time, enabling the
high-resolution region to move per batch without re-instantiating
the model. Replaces the static graph construction in Encoder.__init__
and AssimilatorDecoder.__init__ with a callable that rebuilds when
coordinates change.

Reference:
    Encoder static construction: graph_weather/models/layers/encoder.py
    Decoder static construction: graph_weather/models/layers/assimilator_decoder.py
    Formal spec: .kiro/specs/movable-regional-prototype/requirements.md (Req 1, 2, 7)
"""

from typing import List, Optional, Tuple

import h3
import numpy as np
import torch
from torch_geometric.data import Data


class DynamicGraphBuilder:
    """Builds bipartite and latent graphs from arbitrary lat/lon coordinates.

    Unlike the static graph baked in Encoder.__init__, this class constructs
    graphs at call time. When the same coordinate list object is passed again,
    it returns the previously built graphs without rebuilding (Req 1.4).

    The encoder graph uses 1-to-1 edges (each lat/lon → its H3 cell).
    The decoder graph uses 1-to-many edges (each lat/lon → ring-1 neighborhood).
    The latent graph connects only the H3 cells used by the regional coordinates.

    Args:
        resolution: H3 resolution level for the latent mesh.
    """

    def __init__(self, resolution: int = 2):
        """Initialize with the given H3 resolution level."""
        self.resolution = resolution
        self._prev_lat_lons: Optional[List[Tuple[float, float]]] = None
        self._cached_encoder_graph: Optional[Data] = None
        self._cached_decoder_graph: Optional[Data] = None
        self._cached_latent_graph: Optional[Data] = None
        self._cached_h3_indices: Optional[List[int]] = None

    def _validate(self, lat_lons: List[Tuple[float, float]]) -> None:
        """Validate coordinate list. Raises ValueError on bad input."""
        if not lat_lons:
            raise ValueError(
                "lat_lons must not be empty. Provide at least one coordinate."
            )
        for i, (lat, lon) in enumerate(lat_lons):
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Coordinate {i}: latitude {lat} is outside [-90, 90]."
                )

    def _assign_h3_cells(
        self, lat_lons: List[Tuple[float, float]]
    ) -> Tuple[List[str], List[str], dict]:
        """Map each coordinate to an H3 cell and build index mappings.

        Returns:
            h3_cells: H3 cell ID per coordinate (same order as lat_lons).
            unique_cells: Sorted list of unique H3 cell IDs.
            cell_to_idx: Mapping from H3 cell ID to integer index.
        """
        h3_cells = [
            h3.latlng_to_cell(lat, lon, self.resolution)
            for lat, lon in lat_lons
        ]
        unique_cells = sorted(set(h3_cells))
        cell_to_idx = {cell: i for i, cell in enumerate(unique_cells)}
        return h3_cells, unique_cells, cell_to_idx

    def build_encoder_graph(
        self, lat_lons: List[Tuple[float, float]]
    ) -> Tuple[Data, List[int]]:
        """Build the lat/lon → H3 bipartite graph for encoding.

        Each lat/lon node gets exactly one edge to its assigned H3 cell.
        Edge attributes are [sin(d), cos(d)] where d is the great-circle
        distance in radians, matching encoder.py lines 87-92.

        Args:
            lat_lons: Regional coordinates as (lat, lon) pairs.

        Returns:
            graph: Data with edge_index [2, N] and edge_attr [N, 2].
            h3_indices: Sorted unique H3 cell indices for embedding lookup.
        """
        h3_cells, unique_cells, cell_to_idx = self._assign_h3_cells(lat_lons)
        num_coords = len(lat_lons)

        edge_sources = []
        edge_targets = []
        edge_attrs = []

        for node_idx, (coord, cell) in enumerate(zip(lat_lons, h3_cells)):
            edge_sources.append(node_idx)
            edge_targets.append(num_coords + cell_to_idx[cell])
            dist = h3.great_circle_distance(
                coord, h3.cell_to_latlng(cell), unit="rads"
            )
            edge_attrs.append([np.sin(dist), np.cos(dist)])

        edge_index = torch.tensor(
            [edge_sources, edge_targets], dtype=torch.long
        )
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # Map unique H3 cells to global indices for embedding table lookup
        all_h3 = sorted(
            h3.uncompact_cells(h3.get_res0_cells(), self.resolution)
        )
        global_h3_map = {cell: i for i, cell in enumerate(all_h3)}
        h3_indices = [global_h3_map[cell] for cell in unique_cells]

        return Data(edge_index=edge_index, edge_attr=edge_attr), h3_indices

    def build_decoder_graph(
        self, lat_lons: List[Tuple[float, float]]
    ) -> Data:
        """Build the H3 → lat/lon bipartite graph for decoding.

        Each lat/lon node connects to its assigned H3 cell AND its
        ring-1 neighbors (~7 edges per node). This richer connectivity
        enables interpolation back to arbitrary positions, matching
        assimilator_decoder.py lines 92-101.

        Args:
            lat_lons: Regional coordinates as (lat, lon) pairs.

        Returns:
            Data with edge_index [2, E] and edge_attr [E, 2].
        """
        h3_cells, unique_cells, cell_to_idx = self._assign_h3_cells(lat_lons)

        # Build a mapping that includes ring-1 neighbors of all assigned cells
        all_neighborhood_cells = set()
        for cell in unique_cells:
            all_neighborhood_cells.update(h3.grid_disk(cell, 1))
        all_neighborhood_cells = sorted(all_neighborhood_cells)
        neighborhood_to_idx = {
            cell: i for i, cell in enumerate(all_neighborhood_cells)
        }

        edge_sources = []
        edge_targets = []
        edge_attrs = []

        for node_idx, (coord, cell) in enumerate(zip(lat_lons, h3_cells)):
            neighbors = h3.grid_disk(cell, 1)
            for h in neighbors:
                h_idx = neighborhood_to_idx[h]
                edge_sources.append(h_idx)
                edge_targets.append(len(all_neighborhood_cells) + node_idx)
                dist = h3.great_circle_distance(
                    coord, h3.cell_to_latlng(h), unit="rads"
                )
                edge_attrs.append([np.sin(dist), np.cos(dist)])

        edge_index = torch.tensor(
            [edge_sources, edge_targets], dtype=torch.long
        )
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return Data(edge_index=edge_index, edge_attr=edge_attr)

    def build_latent_graph(self, unique_cells: List[str]) -> Data:
        """Build the H3 ↔ H3 neighbor connectivity for message passing.

        Only connects H3 cells that the regional coordinates map to,
        NOT the full global mesh. Each cell connects to its ring-1
        neighbors (itself + 6 neighbors), matching encoder.py lines 244-268.

        Args:
            unique_cells: Sorted unique H3 cell IDs from the regional coords.

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

        edge_index = torch.tensor(
            [edge_sources, edge_targets], dtype=torch.long
        )
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return Data(edge_index=edge_index, edge_attr=edge_attr)

    def __call__(
        self, lat_lons: List[Tuple[float, float]]
    ) -> Tuple[Data, Data, Data, List[int]]:
        """Build or return cached encoder, decoder, and latent graphs.

        Args:
            lat_lons: Regional coordinates as (lat, lon) pairs.

        Returns:
            encoder_graph: Bipartite lat/lon → H3 (1-to-1 edges).
            decoder_graph: Bipartite H3 → lat/lon (ring-1 edges).
            latent_graph: H3 ↔ H3 regional mesh connectivity.
            h3_indices: Global H3 cell indices for embedding lookup.
        """
        if lat_lons is self._prev_lat_lons:
            return (
                self._cached_encoder_graph,
                self._cached_decoder_graph,
                self._cached_latent_graph,
                self._cached_h3_indices,
            )

        self._validate(lat_lons)

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
