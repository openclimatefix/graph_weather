"""Tests for DynamicGraphBuilder.

Covers graph construction, caching, validation, and the encoder/decoder
topology asymmetry. Uses synthetic coordinates matching the patterns in
tests/test_model.py.
"""

import pytest
import torch
from torch_geometric.data import Data

from graph_weather.models.layers.dynamic_graph_builder import DynamicGraphBuilder


@pytest.fixture
def builder():
    return DynamicGraphBuilder(resolution=2)


@pytest.fixture
def small_region():
    """A small 5x5 lat/lon patch centered on London."""
    lat_lons = []
    for lat in range(50, 55):
        for lon in range(-2, 3):
            lat_lons.append((float(lat), float(lon)))
    return lat_lons


class TestEncoderGraph:
    def test_shape(self, builder, small_region):
        graph, h3_indices = builder.build_encoder_graph(small_region)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.dtype == torch.long
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
        assert graph.edge_attr.shape[1] == 2

    def test_one_edge_per_node(self, builder, small_region):
        graph, _ = builder.build_encoder_graph(small_region)
        assert graph.edge_index.shape[1] == len(small_region)

    def test_h3_indices_returned(self, builder, small_region):
        _, h3_indices = builder.build_encoder_graph(small_region)
        assert len(h3_indices) > 0
        assert all(isinstance(i, int) for i in h3_indices)
        # Must be fewer unique H3 cells than coordinates
        assert len(h3_indices) <= len(small_region)


class TestDecoderGraph:
    def test_more_edges_than_encoder(self, builder, small_region):
        enc_graph, _ = builder.build_encoder_graph(small_region)
        dec_graph = builder.build_decoder_graph(small_region)
        # Decoder uses ring-1 neighborhood: ~7 edges per node vs 1
        assert dec_graph.edge_index.shape[1] > enc_graph.edge_index.shape[1]

    def test_shape(self, builder, small_region):
        graph = builder.build_decoder_graph(small_region)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.dtype == torch.long
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
        assert graph.edge_attr.shape[1] == 2


class TestLatentGraph:
    def test_scoped_to_region(self, builder, small_region):
        """Latent graph has far fewer nodes than the full global mesh."""
        import h3 as h3_lib

        h3_cells = [
            h3_lib.latlng_to_cell(lat, lon, builder.resolution)
            for lat, lon in small_region
        ]
        unique_cells = sorted(set(h3_cells))
        graph = builder.build_latent_graph(unique_cells)

        total_global = h3_lib.get_num_cells(builder.resolution)
        # The latent graph node count (derived from unique_cells) is much
        # smaller than the full global mesh
        assert len(unique_cells) < total_global
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]

    def test_self_loops_present(self, builder, small_region):
        """grid_disk(cell, 1) includes the cell itself, so self-loops exist."""
        import h3 as h3_lib

        h3_cells = [
            h3_lib.latlng_to_cell(lat, lon, builder.resolution)
            for lat, lon in small_region
        ]
        unique_cells = sorted(set(h3_cells))
        graph = builder.build_latent_graph(unique_cells)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        self_loops = (src == dst).sum().item()
        assert self_loops == len(unique_cells)


class TestCaching:
    def test_same_object_returns_cached(self, builder, small_region):
        result1 = builder(small_region)
        result2 = builder(small_region)
        # Same list object → same graph objects (identity check)
        assert result1[0] is result2[0]
        assert result1[1] is result2[1]
        assert result1[2] is result2[2]

    def test_different_coords_rebuild(self, builder, small_region):
        result1 = builder(small_region)
        different_region = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        result2 = builder(different_region)
        assert result1[0] is not result2[0]


class TestValidation:
    def test_empty_list_raises(self, builder):
        with pytest.raises(ValueError, match="must not be empty"):
            builder([])

    def test_invalid_latitude_raises(self, builder):
        with pytest.raises(ValueError, match="latitude"):
            builder([(91.0, 0.0)])

    def test_negative_invalid_latitude_raises(self, builder):
        with pytest.raises(ValueError, match="latitude"):
            builder([(-91.0, 0.0)])

    def test_valid_boundary_latitudes(self, builder):
        """Latitudes exactly at -90 and 90 should work."""
        result = builder([(-90.0, 0.0), (90.0, 180.0)])
        assert result[0].edge_index.shape[1] == 2


class TestEdgeAttributes:
    def test_edge_attr_matches_edge_count(self, builder, small_region):
        enc, _ = builder.build_encoder_graph(small_region)
        dec = builder.build_decoder_graph(small_region)
        assert enc.edge_attr.shape[0] == enc.edge_index.shape[1]
        assert dec.edge_attr.shape[0] == dec.edge_index.shape[1]

    def test_edge_attr_values_bounded(self, builder, small_region):
        """sin/cos values are always in [-1, 1]."""
        enc, _ = builder.build_encoder_graph(small_region)
        assert enc.edge_attr.abs().max() <= 1.0


class TestCallInterface:
    def test_returns_four_items(self, builder, small_region):
        result = builder(small_region)
        assert len(result) == 4
        enc_graph, dec_graph, lat_graph, h3_indices = result
        assert isinstance(enc_graph, Data)
        assert isinstance(dec_graph, Data)
        assert isinstance(lat_graph, Data)
        assert isinstance(h3_indices, list)

    def test_round_trip_ordering(self, builder, small_region):
        """Encoder source nodes should be ordered 0..N-1 matching input."""
        enc_graph, _, _, _ = builder(small_region)
        sources = enc_graph.edge_index[0].tolist()
        assert sources == list(range(len(small_region)))
