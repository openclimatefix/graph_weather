"""Generate hexagonal global grid using Uber's H3 library."""
import h3
import numpy as np
import torch
from torch_geometric.data import Data


def generate_hexagonal_grid(resolution: int = 2) -> np.ndarray:
    """Generate hexagonal global grid using Uber's H3 library.

    Args:
        resolution: H3 resolution level

    Returns:
        Hexagonal grid
    """
    base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
    base_h3_map = {h_i: i for i, h_i in enumerate(base_h3_grid)}
    return np.array(base_h3_grid), base_h3_map


def generate_h3_mapping(lat_lons: list, resolution: int = 2) -> dict:
    """Generate mapping from lat/lon to h3 index.

    Args:
        lat_lons: List of (lat,lon) points
        resolution: H3 resolution level
    """
    num_latlons = len(lat_lons)
    base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
    base_h3_map = {h_i: i for i, h_i in enumerate(base_h3_grid)}
    h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
    h3_mapping = {}
    h_index = len(base_h3_grid)
    for h in base_h3_grid:
        if h not in h3_mapping:
            h_index -= 1
            h3_mapping[h] = h_index + num_latlons
    # Now have the h3 grid mapping, the bipartite graph of edges connecting lat/lon to h3 nodes
    # Should have vertical and horizontal difference
    h3_distances = []
    for idx, h3_point in enumerate(h3_grid):
        lat_lon = lat_lons[idx]
        distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit="rads")
        h3_distances.append([np.sin(distance), np.cos(distance)])
    h3_distances = torch.tensor(h3_distances, dtype=torch.float)
    return base_h3_map, h3_mapping, h3_distances


def generate_latent_h3_graph(base_h3_map: dict, base_h3_grid: dict) -> torch.Tensor:
    """Generate latent h3 graph.

    Args:
        base_h3_map: Mapping from h3 index to index in latent graph
        h3_mapping: Mapping from lat/lon to h3 index
        h3_distances: Distances between lat/lon and h3 index

    Returns:
        Latent h3 graph
    """
    # Get connectivity of the graph
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    for h3_index in base_h3_grid:
        h_points = h3.k_ring(h3_index, 1)
        for h in h_points:  # Already includes itself
            distance = h3.point_dist(h3.h3_to_geo(h3_index), h3.h3_to_geo(h), unit="rads")
            edge_attrs.append([np.sin(distance), np.cos(distance)])
            edge_sources.append(base_h3_map[h3_index])
            edge_targets.append(base_h3_map[h])
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
    # Use heterogeneous graph as input and output dims are not same for the encoder
    # Because uniform grid now, don't need edge attributes as they are all the same
    return Data(edge_index=edge_index, edge_attr=edge_attrs)
