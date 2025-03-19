import pytest
import numpy as np
import networkx as nx
from graph_weather.data.anemoi_graph_gen import AnemoiGraphAdapter, AnemoiGrid

def test_Anemoi_grid_vertex_properties():
    """
    Verify that AnemoiGrid produces a valid set of vertices.
    - Vertices should be a 2D numpy array of shape (N, 3).
    """
    grid = AnemoiGrid(grid_resolution=128)
    vertices = grid.get_vertices()
    assert vertices.ndim == 2, "Vertices should be a 2D array."
    assert vertices.shape[1] == 3, "Each vertex should have 3 coordinates."

def test_Anemoi_grid_latlon_range():
    """
    Ensure latitude and longitude values are within valid ranges:
      - Latitude in [-90, 90]
      - Longitude in [-180, 180]
    """
    grid = AnemoiGrid(grid_resolution=128)
    latlon = grid.get_latlon_coordinates()
    for lat, lon in latlon:
        assert -90 <= lat <= 90, f"Latitude {lat} out of range."
        assert -180 <= lon <= 180, f"Longitude {lon} out of range."

def test_Anemoi_grid_graph_connectivity():
    """
    Verify that the graph built by AnemoiGrid is fully connected.
    """
    grid = AnemoiGrid(grid_resolution=128)
    G = grid.get_graph()
    assert nx.is_connected(G), "Graph should be connected."

def test_Anemoi_grid_adjacency_symmetry():
    """
    Check that the adjacency information is symmetric.
    """
    grid = AnemoiGrid(grid_resolution=128)
    src, dst = grid.get_adjacency_info()
    # Build a dictionary mapping each node to its neighbors
    adj = {}
    for u, v in zip(src, dst):
        adj.setdefault(u, []).append(v)
    for u in adj:
        for v in adj[u]:
            assert u in adj.get(v, []), f"Edge ({u}, {v}) is not symmetric."

def test_sliding_window_attention_indices():
    """
    Verify that sliding window attention indices are generated.
    Each window should be a list of node indices with at least half the window size.
    """
    grid = AnemoiGrid(grid_resolution=128)
    windows = grid.get_sliding_window_attention_indices(window_size=40)
    assert isinstance(windows, list), "Attention windows should be a list."
    assert len(windows) > 0, "There should be at least one attention window."
    for window in windows:
        assert isinstance(window, list), "Each attention window should be a list of indices."
        assert len(window) >= 40 // 2, "Attention window size is too small."

def test_adapter_node_features():
    """
    Test that AnemoiGraphAdapter correctly maps input data to node features.
    Uses a small, constant input to avoid scalability issues.
    """
    adapter = AnemoiGraphAdapter(resolution=128)
    # Use a small constant array with shape (3, 10, 20)
    dummy_data = np.ones((3, 10, 20))
    features = adapter.get_node_features(dummy_data, variable_idx=0)
    assert features.ndim == 1, "Node features should be a 1D array."
    assert features.shape[0] == adapter.n_nodes, "Feature vector length should equal number of nodes."

def test_prepare_model_inputs():
    """
    Verify that prepare_model_inputs returns a dictionary with the correct keys and shapes.
    Dummy input data arrays use a smaller shape (3, 10, 20) for scalability.
    Expected keys: 'node_features', 'edge_index', 'attention_masks', 'positions'
    """
    adapter = AnemoiGraphAdapter(resolution=128)
    data = {
        'temperature': np.ones((3, 10, 20)),
        'pressure': np.ones((3, 10, 20)),
        'humidity': np.ones((3, 10, 20))
    }
    inputs = adapter.prepare_model_inputs(data)
    expected_keys = {'node_features', 'edge_index', 'attention_masks', 'positions'}
    assert expected_keys.issubset(set(inputs.keys())), "Missing expected keys in model inputs."
    
    nf = inputs['node_features']
    assert nf.ndim == 2, "node_features should be a 2D array."
    assert nf.shape[0] == adapter.n_nodes, "First dimension of node_features must equal number of nodes."
    
    ei = inputs['edge_index']
    assert ei.ndim == 2 and ei.shape[0] == 2, "edge_index should have shape (2, num_edges)."
    
    am = inputs['attention_masks']
    assert am.ndim == 2 and am.shape[1] == adapter.n_nodes, "attention_masks shape is incorrect."
    
    pos = inputs['positions']
    assert pos.ndim == 2 and pos.shape[1] == 3, "positions should have shape (n_nodes, 3)."

def test_process_model_outputs():
    """
    Test that process_model_outputs regrids model outputs to a 2D grid.
    The output dictionary should contain arrays of shape (180, 360) for each variable.
    """
    adapter = AnemoiGraphAdapter(resolution=128)
    dummy_outputs = np.random.rand(adapter.n_nodes, 3)  # Assume 3 output variables
    processed = adapter.process_model_outputs(dummy_outputs)
    
    for var_idx in range(3):
        key = f'var_{var_idx}'
        assert key in processed, f"Missing output for variable {var_idx}."
        grid = processed[key]
        assert grid.shape == (180, 360), f"Regridded output shape for {key} should be (180, 360)."
