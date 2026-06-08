"""Tests for dynamic H3 graph construction."""

import h3
import pytest
import torch

from graph_weather.models.layers.dynamic_graph_builder import DynamicGraphBuilder
from graph_weather.utils import validate_lat_lons


def _small_region():
    """Create a small set of lat/lon coordinates in a bounding box."""
    lat_lons = []
    for lat in range(50, 55):
        for lon in range(-2, 3):
            lat_lons.append((float(lat), float(lon)))
    return lat_lons


def test_encoder_graph():
    """Test encoder graph shape and attributes."""
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    graph, h3_indices = builder.build_encoder_graph(lat_lons)

    assert graph.edge_index.shape == (2, 25)
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_index[0].tolist() == list(range(25))
    assert graph.edge_index[1].min().item() >= 25
    assert graph.edge_attr.shape == (25, 2)
    assert graph.edge_attr.abs().max() <= 1.0

    total_h3 = h3.get_num_cells(2)
    assert all(0 <= idx < total_h3 for idx in h3_indices)


def test_decoder_graph():
    """Test decoder graph shape and attributes."""
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    graph = builder.build_decoder_graph(lat_lons)

    assert graph.edge_index.shape == (2, 175)
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_attr.shape == (175, 2)
    assert graph.edge_attr.abs().max() <= 1.0


def test_latent_graph():
    """Test latent graph shape and self-loops."""
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    h3_cells = [h3.latlng_to_cell(lat, lon, 2) for lat, lon in lat_lons]
    unique_cells = sorted(set(h3_cells))

    assert len(unique_cells) == 5

    graph = builder.build_latent_graph(unique_cells)

    assert graph.edge_index.shape == (2, 19)
    assert graph.edge_attr.shape == (19, 2)

    src, dst = graph.edge_index[0], graph.edge_index[1]
    self_loops = (src == dst).sum().item()
    assert self_loops == 5


def test_builder_caching():
    """Test builder caching of graphs and H3 indices."""
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()

    res1 = builder(lat_lons)
    res2 = builder(lat_lons)

    assert res1[0] is res2[0]
    assert res1[1] is res2[1]
    assert res1[2] is res2[2]
    assert res1[3] is res2[3]

    different = [(0.0, 0.0), (1.0, 1.0)]
    res3 = builder(different)
    assert res1[0] is not res3[0]


def test_validation_ranges():
    """Test coordinate validation ranges."""
    builder = DynamicGraphBuilder(resolution=2)

    with pytest.raises(ValueError, match="must not be empty"):
        builder([])

    with pytest.raises(ValueError, match="latitude"):
        builder([(91.0, 0.0)])

    with pytest.raises(ValueError, match="latitude"):
        builder([(-91.0, 0.0)])

    res = builder([(-90.0, 0.0), (90.0, 180.0)])
    assert res[0].edge_index.shape == (2, 2)


def test_validate_lat_lons_standalone():
    """Test standalone lat/lon validation function."""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_lat_lons([])
    with pytest.raises(ValueError, match="latitude"):
        validate_lat_lons([(91.0, 0.0)])
    validate_lat_lons([(0.0, 0.0), (45.0, 90.0)])
