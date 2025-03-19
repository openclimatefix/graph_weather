"""
This file contains tests for the IcosahedralGrid class and its associated functions.
The tests validate that the generated canonical icosahedron has:
    - 12 vertices, 20 faces, and 30 unique edges,
    - correct subdivision behavior,
    - valid geographic (latitude, longitude) coordinates,
    - a consistent adjacency list, and
    - full graph connectivity.
"""
import pytest
import numpy as np
import networkx as nx

from graph_weather.data.icosahedral_graph_gen import IcosahedralGrid, create_icosahedral_graph,get_grid_metadata


def test_base_icosahedron_counts():
    """
    Validate that the base icosahedron (with subdivision_level=0) conforms to canonical properties.
    
    Expected:
      - 12 vertices
      - 20 faces
      - 30 unique edges (derived from 20 faces, each contributing 3 edges, with each edge shared by 2 faces)
    """
    grid = IcosahedralGrid(subdivision_level=0)
    G = grid.get_graph()
    
    # Verify the counts of vertices, edges, and faces
    assert G.number_of_nodes() == 12, "Base icosahedron should have 12 vertices."
    assert G.number_of_edges() == 30, "Base icosahedron should have 30 edges."
    assert len(grid.faces) == 20, "Base icosahedron should have 20 faces."

def test_subdivided_icosahedron_counts():
    """
    Ensure that applying a subdivision (increasing the subdivision_level) results in more vertices and edges.
    
    Rationale:
      Subdivision should refine the grid, thus increasing the resolution.
    """
    grid_1 = IcosahedralGrid(subdivision_level=1)
    G_1 = grid_1.get_graph()
    
    # After subdivision, the graph should have more than 12 vertices and more than 30 edges.
    assert G_1.number_of_nodes() > 12, "Subdivision should increase the number of vertices."
    assert G_1.number_of_edges() > 30, "Subdivision should increase the number of edges."

def test_latlon_range():
    """
    Verify that the conversion from Cartesian coordinates to latitude/longitude produces valid ranges.
    
    Expected ranges:
      - Latitude: between -90 and 90 degrees
      - Longitude: between -180 and 180 degrees
    """
    grid = IcosahedralGrid(subdivision_level=0)
    latlon_coords = grid.get_latlon_coordinates()
    
    # Validate each (lat, lon) pair
    for lat, lon in latlon_coords:
        assert -90.0 <= lat <= 90.0, f"Latitude {lat} out of range."
        assert -180.0 <= lon <= 180.0, f"Longitude {lon} out of range."

def test_adjacency_list():
    """
    Verify that the graph's adjacency list is correctly constructed.
    
    Expected:
      - Exactly 12 entries in the adjacency list (one per vertex)
      - Each vertex's neighbors are in the range [0, 11]
      - Neighbor relationships are symmetric (if A is a neighbor of B, then B must be a neighbor of A)
    """
    grid = IcosahedralGrid(subdivision_level=0)
    adjacency_list = grid.get_adjacency_list()
    
    # Check that there are 12 vertices
    assert len(adjacency_list) == 12, "Adjacency list should have 12 entries."
    
    # Validate each node's neighbor relationships
    for node, neighbors in adjacency_list.items():
        assert 0 <= node < 12, f"Node {node} is out of expected range."
        for nbr in neighbors:
            assert 0 <= nbr < 12, f"Neighbor {nbr} is out of expected range."
        # Ensure reciprocity in neighbor relationships
        for nbr in neighbors:
            assert node in adjacency_list[nbr], "Adjacency is not symmetric."

def test_get_grid_metadata():
    """
    Check that the metadata extraction function returns correct information about the icosahedral grid.
    
    Expected metadata keys:
      - num_vertices: should equal 12
      - num_edges: should equal 30
      - positions: a NumPy array with shape (12, 3)
      - latlon: list of 12 (lat, lon) pairs
      - edges: a list of 30 unique edges
    """
    grid = IcosahedralGrid(subdivision_level=0)
    G = grid.get_graph()
    meta = get_grid_metadata(G)
    
    # Verify all expected keys are present in the metadata
    for key in ['num_vertices', 'num_edges', 'positions', 'latlon', 'edges']:
        assert key in meta, f"Metadata missing expected key: {key}"
    
    # Validate metadata values
    assert meta['num_vertices'] == 12, "Base icosahedron should have 12 vertices."
    assert meta['num_edges'] == 30, "Base icosahedron should have 30 edges."
    assert len(meta['positions']) == 12, "Positions array should match number of vertices."
    assert len(meta['latlon']) == 12, "Lat/Lon list should match number of vertices."
    assert len(meta['edges']) == 30, "Edge list should match number of edges."

def test_create_icosahedral_graph_function():
    """
    Verify that the convenience function 'create_icosahedral_graph()' produces a valid graph.
    
    The function should return a NetworkX graph with:
      - 12 vertices and 30 edges for level=0
      - A higher number of vertices and edges for level=1
    """
    G0 = create_icosahedral_graph(level=0)
    assert G0.number_of_nodes() == 12, "create_icosahedral_graph(level=0) should have 12 nodes."
    assert G0.number_of_edges() == 30, "create_icosahedral_graph(level=0) should have 30 edges."
    
    G1 = create_icosahedral_graph(level=1)
    assert G1.number_of_nodes() > 12, "create_icosahedral_graph(level=1) should have more than 12 nodes."
    assert G1.number_of_edges() > 30, "create_icosahedral_graph(level=1) should have more than 30 edges."

def test_vertices_on_unit_sphere():
    """
    Ensure that all vertices lie on the surface of a unit sphere.
    
    This is confirmed by checking that the Euclidean norm of each vertex equals 1 (within a small tolerance).
    """
    grid = IcosahedralGrid(subdivision_level=0)
    vertices = grid.get_vertices()
    
    for i, v in enumerate(vertices):
        norm = np.linalg.norm(v)
        assert np.allclose(norm, 1.0, atol=1e-6), f"Vertex {v} is not on the unit sphere (norm = {norm})."

def test_graph_connectivity():
    """
    Validate that the generated graph is fully connected.
    
    A fully connected graph means there is a path between any two vertices.
    """
    grid = IcosahedralGrid(subdivision_level=0)
    G = grid.get_graph()
    assert nx.is_connected(G), "The generated graph should be connected."