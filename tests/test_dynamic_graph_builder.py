import h3
import pytest
import torch
from torch_geometric.data import Data

from graph_weather.models.layers.dynamic_graph_builder import DynamicGraphBuilder
from graph_weather.utils import validate_lat_lons


def _small_region():
    """5x5 lat/lon patch around London."""
    lat_lons = []
    for lat in range(50, 55):
        for lon in range(-2, 3):
            lat_lons.append((float(lat), float(lon)))
    return lat_lons


def test_encoder_graph_shape():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    graph, h3_indices = builder.build_encoder_graph(lat_lons)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    assert graph.edge_attr.shape[1] == 2


def test_encoder_one_edge_per_node():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    graph, _ = builder.build_encoder_graph(lat_lons)
    assert graph.edge_index.shape[1] == len(lat_lons)


def test_encoder_h3_indices():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    _, h3_indices = builder.build_encoder_graph(lat_lons)
    assert len(h3_indices) > 0
    assert all(isinstance(i, int) for i in h3_indices)
    assert len(h3_indices) <= len(lat_lons)


def test_decoder_more_edges_than_encoder():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    enc_graph, _ = builder.build_encoder_graph(lat_lons)
    dec_graph = builder.build_decoder_graph(lat_lons)
    assert dec_graph.edge_index.shape[1] > enc_graph.edge_index.shape[1]


def test_decoder_graph_shape():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    graph = builder.build_decoder_graph(lat_lons)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    assert graph.edge_attr.shape[1] == 2


def test_latent_graph_scoped_to_region():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    h3_cells = [h3.latlng_to_cell(lat, lon, 2) for lat, lon in lat_lons]
    unique_cells = sorted(set(h3_cells))
    graph = builder.build_latent_graph(unique_cells)

    total_global = h3.get_num_cells(2)
    assert len(unique_cells) < total_global
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]


def test_latent_graph_self_loops():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    h3_cells = [h3.latlng_to_cell(lat, lon, 2) for lat, lon in lat_lons]
    unique_cells = sorted(set(h3_cells))
    graph = builder.build_latent_graph(unique_cells)

    src, dst = graph.edge_index[0], graph.edge_index[1]
    self_loops = (src == dst).sum().item()
    assert self_loops == len(unique_cells)


def test_caching_same_object():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    result1 = builder(lat_lons)
    result2 = builder(lat_lons)
    assert result1[0] is result2[0]
    assert result1[1] is result2[1]
    assert result1[2] is result2[2]


def test_caching_different_coords_rebuild():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    result1 = builder(lat_lons)
    different = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    result2 = builder(different)
    assert result1[0] is not result2[0]


def test_empty_list_raises():
    builder = DynamicGraphBuilder(resolution=2)
    with pytest.raises(ValueError, match="must not be empty"):
        builder([])


def test_invalid_latitude_raises():
    builder = DynamicGraphBuilder(resolution=2)
    with pytest.raises(ValueError, match="latitude"):
        builder([(91.0, 0.0)])


def test_negative_invalid_latitude_raises():
    builder = DynamicGraphBuilder(resolution=2)
    with pytest.raises(ValueError, match="latitude"):
        builder([(-91.0, 0.0)])


def test_valid_boundary_latitudes():
    builder = DynamicGraphBuilder(resolution=2)
    result = builder([(-90.0, 0.0), (90.0, 180.0)])
    assert result[0].edge_index.shape[1] == 2


def test_edge_attr_matches_edge_count():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    enc, _ = builder.build_encoder_graph(lat_lons)
    dec = builder.build_decoder_graph(lat_lons)
    assert enc.edge_attr.shape[0] == enc.edge_index.shape[1]
    assert dec.edge_attr.shape[0] == dec.edge_index.shape[1]


def test_edge_attr_values_bounded():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    enc, _ = builder.build_encoder_graph(lat_lons)
    assert enc.edge_attr.abs().max() <= 1.0


def test_call_returns_four_items():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    result = builder(lat_lons)
    assert len(result) == 4
    enc_graph, dec_graph, lat_graph, h3_indices = result
    assert isinstance(enc_graph, Data)
    assert isinstance(dec_graph, Data)
    assert isinstance(lat_graph, Data)
    assert isinstance(h3_indices, list)


def test_encoder_source_ordering():
    builder = DynamicGraphBuilder(resolution=2)
    lat_lons = _small_region()
    enc_graph, _, _, _ = builder(lat_lons)
    sources = enc_graph.edge_index[0].tolist()
    assert sources == list(range(len(lat_lons)))


def test_validate_lat_lons_standalone():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_lat_lons([])
    with pytest.raises(ValueError, match="latitude"):
        validate_lat_lons([(91.0, 0.0)])
    validate_lat_lons([(0.0, 0.0), (45.0, 90.0)])
