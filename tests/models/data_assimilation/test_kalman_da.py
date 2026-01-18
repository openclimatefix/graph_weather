"""
Comprehensive tests for the Kalman Filter Data Assimilation method.

Tests include functionality for both tensor and graph-based inputs.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
import sys
import os

sys.path.insert(0, os.path.abspath("."))

# Use direct import to avoid package conflicts
import importlib.util

# Add the graph_weather directory to the path to make relative imports work
sys.path.insert(0, os.path.join(os.getcwd(), "graph_weather"))

# Load base module first
spec = importlib.util.spec_from_file_location(
    "data_assimilation_base", "./graph_weather/models/data_assimilation/data_assimilation_base.py"
)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)

# Load kalman module with proper base class injection
spec = importlib.util.spec_from_file_location(
    "kalman_filter_da", "./graph_weather/models/data_assimilation/kalman_filter_da.py"
)
kalman_module = importlib.util.module_from_spec(spec)
kalman_module.DataAssimilationBase = base_module.DataAssimilationBase
kalman_module.EnsembleGenerator = base_module.EnsembleGenerator
kalman_module.Data = Data
kalman_module.HeteroData = getattr(
    __import__("torch_geometric.data", fromlist=["HeteroData"]), "HeteroData", None
)
kalman_module.torch = torch
kalman_module.nn = torch.nn
kalman_module.typing = __import__("typing")
spec.loader.exec_module(kalman_module)

KalmanFilterDA = kalman_module.KalmanFilterDA
EnsembleGenerator = base_module.EnsembleGenerator


def test_kalman_filter_initialization():
    """Test Kalman Filter DA initialization with default and custom configs."""
    # Test default initialization
    kf_da = KalmanFilterDA()
    assert kf_da.ensemble_size == 20
    assert kf_da.inflation_factor == 1.1
    assert kf_da.observation_error_std == 0.1
    assert kf_da.background_error_std == 0.5

    # Test custom initialization
    config = {
        "ensemble_size": 30,
        "inflation_factor": 1.2,
        "observation_error_std": 0.05,
        "background_error_std": 0.3,
        "adaptive_inflation": False,
    }
    kf_da_custom = KalmanFilterDA(config)
    assert kf_da_custom.ensemble_size == 30
    assert kf_da_custom.inflation_factor == 1.2
    assert kf_da_custom.observation_error_std == 0.05
    assert kf_da_custom.background_error_std == 0.3
    assert not kf_da_custom.adaptive_inflation


def test_kalman_tensor_forward():
    """Test Kalman Filter DA forward pass with tensor inputs."""
    kf_da = KalmanFilterDA({"ensemble_size": 10})

    # Create input state and observations
    batch_size = 2
    state_features = 16
    obs_features = 8

    state_in = torch.randn(batch_size, state_features)
    observations = torch.randn(batch_size, obs_features)

    # Forward pass
    result = kf_da(state_in, observations)

    # Check output shape
    assert result.shape == state_in.shape
    assert torch.is_tensor(result)


def test_kalman_tensor_initialize_ensemble():
    """Test ensemble initialization with tensor inputs."""
    kf_da = KalmanFilterDA({"background_error_std": 0.1})

    # Create background state
    batch_size = 3
    state_features = 12
    background_state = torch.randn(batch_size, state_features)
    num_members = 5

    # Initialize ensemble
    ensemble = kf_da.initialize_ensemble(background_state, num_members)

    # Check ensemble shape
    assert ensemble.shape == (batch_size, num_members, state_features)
    # Check that ensemble members are similar but not identical to background
    ensemble_mean = torch.mean(ensemble, dim=1)
    assert torch.allclose(ensemble_mean, background_state, atol=0.2)


def test_kalman_tensor_assimilate():
    """Test assimilation with tensor inputs."""
    kf_da = KalmanFilterDA({"ensemble_size": 8, "observation_error_std": 0.1})

    # Create ensemble and observations
    batch_size = 2
    state_features = 10
    obs_features = 5
    num_members = 8

    ensemble = torch.randn(batch_size, num_members, state_features)
    observations = torch.randn(batch_size, obs_features)

    # Perform assimilation
    updated_ensemble = kf_da.assimilate(ensemble, observations)

    # Check output shape
    assert updated_ensemble.shape == ensemble.shape
    assert torch.is_tensor(updated_ensemble)


def test_kalman_graph_forward():
    """Test Kalman Filter DA forward pass with graph inputs."""
    kf_da = KalmanFilterDA({"ensemble_size": 5})

    # Create graph input
    num_nodes = 8
    node_features = 16
    edge_features = 4

    graph_state = Data(
        x=torch.randn(num_nodes, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 15)),
        edge_attr=torch.randn(15, edge_features) if edge_features > 0 else None,
    )

    # Create observations
    obs_features = 8
    observations = torch.randn(1, obs_features)  # Batch size of 1 for graph

    # Forward pass
    result = kf_da(graph_state, observations)

    # Check output is a graph with same structure
    assert hasattr(result, "x")
    assert result.x.shape == graph_state.x.shape
    assert torch.equal(result.edge_index, graph_state.edge_index)


def test_kalman_graph_initialize_ensemble():
    """Test ensemble initialization with graph inputs."""
    kf_da = KalmanFilterDA({"background_error_std": 0.1})

    # Create background graph state
    num_nodes = 6
    node_features = 10
    background_graph = Data(
        x=torch.randn(num_nodes, node_features), edge_index=torch.randint(0, num_nodes, (2, 10))
    )
    num_members = 4

    # Initialize ensemble
    ensemble = kf_da.initialize_ensemble(background_graph, num_members)

    # Check ensemble has correct structure
    assert hasattr(ensemble, "x")
    # For graphs, ensemble creates [num_nodes, num_members, features] shape for x
    assert ensemble.x.shape == (num_nodes, num_members, node_features)


def test_kalman_graph_assimilate():
    """Test assimilation with graph inputs."""
    kf_da = KalmanFilterDA({"ensemble_size": 6})

    # Create graph ensemble and observations
    num_nodes = 5
    node_features = 8
    obs_features = 6

    graph_ensemble = Data(
        x=torch.randn(num_nodes, 6, node_features),  # [num_nodes, num_members, features]
        edge_index=torch.randint(0, num_nodes, (2, 10)),
    )
    observations = torch.randn(1, obs_features)  # Batch size of 1 for graph

    # Perform assimilation
    updated_ensemble = kf_da.assimilate(graph_ensemble, observations)

    # Check output shape
    assert hasattr(updated_ensemble, "x")
    assert updated_ensemble.x.shape == graph_ensemble.x.shape


def test_kalman_compute_analysis_tensor():
    """Test analysis computation for tensor ensembles."""
    kf_da = KalmanFilterDA()

    # Create ensemble
    batch_size = 3
    num_members = 5
    state_features = 12
    ensemble = torch.randn(batch_size, num_members, state_features)

    # Compute analysis
    analysis = kf_da._compute_analysis(ensemble)

    # Check analysis shape
    assert analysis.shape == (batch_size, state_features)
    # Check that analysis is mean of ensemble
    expected_mean = torch.mean(ensemble, dim=1)
    assert torch.allclose(analysis, expected_mean, atol=1e-5)


def test_kalman_compute_analysis_graph():
    """Test analysis computation for graph ensembles."""
    kf_da = KalmanFilterDA()

    # Create graph ensemble
    num_nodes = 4
    num_members = 3
    node_features = 8
    graph_ensemble = Data(
        x=torch.randn(num_nodes, num_members, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 8)),
    )

    # Compute analysis
    analysis = kf_da._compute_analysis(graph_ensemble)

    # Check analysis has correct structure
    assert hasattr(analysis, "x")
    assert analysis.x.shape == (num_nodes, node_features)
    # Check that analysis x is mean of ensemble x
    expected_mean = torch.mean(graph_ensemble.x, dim=1)
    assert torch.allclose(analysis.x, expected_mean, atol=1e-5)


def test_kalman_different_inflation_modes():
    """Test Kalman Filter with both adaptive and fixed inflation."""
    # Test adaptive inflation
    config_adaptive = {"adaptive_inflation": True, "inflation_factor": 1.1}
    kf_adaptive = KalmanFilterDA(config_adaptive)
    assert kf_adaptive.adaptive_inflation
    assert hasattr(kf_adaptive, "inflation_param")

    # Test fixed inflation
    config_fixed = {"adaptive_inflation": False, "inflation_factor": 1.2}
    kf_fixed = KalmanFilterDA(config_fixed)
    assert not kf_fixed.adaptive_inflation
    assert not hasattr(kf_fixed, "inflation_param")


def test_kalman_error_handling():
    """Test error handling for invalid inputs."""
    kf_da = KalmanFilterDA()

    # Test invalid ensemble type
    with pytest.raises(TypeError):
        kf_da.assimilate("invalid_type", torch.randn(2, 5))


if __name__ == "__main__":
    print("Running Kalman Filter DA tests...")

    test_kalman_filter_initialization()
    print("✓ Initialization test passed")

    test_kalman_tensor_forward()
    print("✓ Tensor forward test passed")

    test_kalman_tensor_initialize_ensemble()
    print("✓ Tensor ensemble initialization test passed")

    test_kalman_tensor_assimilate()
    print("✓ Tensor assimilation test passed")

    test_kalman_graph_forward()
    print("✓ Graph forward test passed")

    test_kalman_graph_initialize_ensemble()
    print("✓ Graph ensemble initialization test passed")

    test_kalman_graph_assimilate()
    print("✓ Graph assimilation test passed")

    test_kalman_compute_analysis_tensor()
    print("✓ Tensor analysis computation test passed")

    test_kalman_compute_analysis_graph()
    print("✓ Graph analysis computation test passed")

    test_kalman_different_inflation_modes()
    print("✓ Inflation modes test passed")

    test_kalman_error_handling()
    print("✓ Error handling test passed")

    print("\n✅ All Kalman Filter DA tests passed!")
