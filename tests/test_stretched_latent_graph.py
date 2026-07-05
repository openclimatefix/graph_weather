"""Tests for the variable-resolution latent graph builder."""

import h3
import torch

from graph_weather.models.layers.stretched_latent_graph import (
    build_variable_resolution_latent_graph,
)
from graph_weather.models.layers.stretched_mesh import build_variable_resolution_mesh

BBOX = (50.0, 55.0, -2.0, 3.0)
COARSE_RES = 2
FINE_RES = 4


def _mesh():
    """A real coarse-globe-plus-fine-region mesh to wire into a graph."""
    return build_variable_resolution_mesh(BBOX, COARSE_RES, FINE_RES)


def _edge_pairs(graph):
    """Directed (source, target) index pairs from a graph's edge_index."""
    return list(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))


def test_seam_has_cross_resolution_edges():
    """At least one edge joins a coarse cell to a fine cell across the seam."""
    cells = _mesh()
    graph = build_variable_resolution_latent_graph(cells)
    res = [h3.get_resolution(c) for c in sorted(cells)]
    cross = [(s, t) for s, t in _edge_pairs(graph) if res[s] != res[t]]
    assert len(cross) > 0


def test_graph_is_connected():
    """Every node reaches every other: the fine region is not a sealed island."""
    cells = _mesh()
    graph = build_variable_resolution_latent_graph(cells)
    n = len(cells)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, t in _edge_pairs(graph):
        parent[find(s)] = find(t)
    roots = {find(i) for i in range(n)}
    assert len(roots) == 1


def test_interior_same_resolution_connected():
    """Two adjacent fine cells inside the region share an edge."""
    cells = _mesh()
    sorted_cells = sorted(cells)
    idx = {c: i for i, c in enumerate(sorted_cells)}
    fine = [c for c in sorted_cells if h3.get_resolution(c) == FINE_RES]
    pair = None
    for c in fine:
        for nb in h3.grid_disk(c, 1):
            if nb != c and nb in idx and h3.get_resolution(nb) == FINE_RES:
                pair = (idx[c], idx[nb])
                break
        if pair:
            break
    graph = build_variable_resolution_latent_graph(cells)
    assert pair in _edge_pairs(graph)


def test_no_self_loops():
    """No cell has an edge to itself."""
    graph = build_variable_resolution_latent_graph(_mesh())
    assert all(s != t for s, t in _edge_pairs(graph))


def test_no_duplicate_edges():
    """Each directed edge appears exactly once."""
    graph = build_variable_resolution_latent_graph(_mesh())
    pairs = _edge_pairs(graph)
    assert len(pairs) == len(set(pairs))


def test_node_count_matches_cells():
    """edge_index references exactly the supplied cells, no more."""
    cells = _mesh()
    graph = build_variable_resolution_latent_graph(cells)
    assert int(graph.edge_index.max()) < len(cells)


def test_edge_attr_on_unit_circle():
    """edge_attr is [E, 2] sin/cos pairs lying on the unit circle."""
    graph = build_variable_resolution_latent_graph(_mesh())
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 2)
    norms = graph.edge_attr[:, 0] ** 2 + graph.edge_attr[:, 1] ** 2
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
