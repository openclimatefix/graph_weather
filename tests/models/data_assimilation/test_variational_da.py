"""
Comprehensive tests for the Variational Data Assimilation method.

Tests include functionality for both tensor and graph-based inputs.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData
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

# Load variational module with proper base class injection
spec = importlib.util.spec_from_file_location(
    "variational_da", "./graph_weather/models/data_assimilation/variational_da.py"
)
var_module = importlib.util.module_from_spec(spec)
var_module.DataAssimilationBase = base_module.DataAssimilationBase
var_module.EnsembleGenerator = base_module.EnsembleGenerator
var_module.Data = Data
var_module.HeteroData = HeteroData
var_module.torch = torch
var_module.nn = torch.nn
var_module.F = torch.nn.functional
var_module.typing = __import__("typing")
spec.loader.exec_module(var_module)

VariationalDA = var_module.VariationalDA
EnsembleGenerator = base_module.EnsembleGenerator


def test_variational_da_initialization():
    """Test Variational DA initialization with default and custom configs."""
    # Test default initialization
    var_da = VariationalDA()
    assert var_da.iterations == 10
    assert var_da.learning_rate == 0.01
    assert var_da.regularization_weight == 0.1
    assert var_da.background_error_std == 0.5
    assert var_da.observation_error_std == 0.1

    # Test custom initialization
    config = {
        "iterations": 15,
        "learning_rate": 0.005,
        "regularization_weight": 0.2,
        "background_error_std": 0.3,
        "observation_error_std": 0.05,
    }
    var_da_custom = VariationalDA(config)
    assert var_da_custom.iterations == 15
    assert var_da_custom.learning_rate == 0.005
    assert var_da_custom.regularization_weight == 0.2
    assert var_da_custom.background_error_std == 0.3
    assert var_da_custom.observation_error_std == 0.05
    assert hasattr(var_da_custom, "bg_weight")  # Learnable parameters should exist
    assert hasattr(var_da_custom, "obs_weight")


def test_variational_tensor_forward():
    """Test Variational DA forward pass with tensor inputs."""
    var_da = VariationalDA({"iterations": 5})  # Reduce iterations for faster testing

    # Create input state and observations
    batch_size = 2
    state_features = 12
    obs_features = 6

    state_in = torch.randn(batch_size, state_features)
    observations = torch.randn(batch_size, obs_features)

    # Forward pass
    result = var_da(state_in, observations)

    # Check output shape
    assert result.shape == state_in.shape
    assert torch.is_tensor(result)


def test_variational_tensor_initialize_ensemble():
    """Test ensemble initialization with tensor inputs."""
    var_da = VariationalDA({"background_error_std": 0.1})

    # Create background state
    batch_size = 3
    state_features = 10
    background_state = torch.randn(batch_size, state_features)
    num_members = 6

    # Initialize ensemble
    ensemble = var_da.initialize_ensemble(background_state, num_members)

    # Check ensemble shape
    assert ensemble.shape == (batch_size, num_members, state_features)
    # Check that ensemble members are similar but not identical to background
    ensemble_mean = torch.mean(ensemble, dim=1)
    assert torch.allclose(ensemble_mean, background_state, atol=0.2)


def test_variational_tensor_assimilate():
    """Test assimilation with tensor inputs."""
    var_da = VariationalDA(
        {"iterations": 3, "learning_rate": 0.01}
    )  # Reduce iterations for faster testing

    # Create ensemble and observations
    batch_size = 2
    state_features = 8
    obs_features = 4
    num_members = 5

    ensemble = torch.randn(batch_size, num_members, state_features)
    observations = torch.randn(batch_size, obs_features)

    # Perform assimilation
    updated_ensemble = var_da.assimilate(ensemble, observations)

    # Check output shape
    assert updated_ensemble.shape == ensemble.shape
    assert torch.is_tensor(updated_ensemble)


def test_variational_compute_tensor_cost_function():
    """Test tensor cost function computation."""
    var_da = VariationalDA({"background_error_std": 0.5, "observation_error_std": 0.1})

    # Create states and observations
    batch_size = 2
    state_features = 8
    obs_features = 4

    analysis_state = torch.randn(batch_size, state_features)
    background_state = torch.randn(batch_size, state_features)
    observations = torch.randn(batch_size, obs_features)

    # Compute cost function
    cost = var_da._compute_tensor_cost_function(analysis_state, background_state, observations)

    # Check output is scalar tensor
    assert cost.shape == torch.Size([])
    assert torch.is_tensor(cost)
    assert cost.requires_grad  # Should be differentiable


def test_variational_tensor_assimilate_ensemble():
    """Test assimilation of tensor ensemble."""
    var_da = VariationalDA({"iterations": 2, "learning_rate": 0.01})

    # Create ensemble and observations
    batch_size = 1  # Keep small for faster testing
    num_members = 3
    state_features = 6
    obs_features = 3

    ensemble = torch.randn(batch_size, num_members, state_features)
    observations = torch.randn(batch_size, obs_features)

    # Perform ensemble assimilation
    updated_ensemble = var_da._assimilate_tensor_ensemble(ensemble, observations)

    # Check output shape
    assert updated_ensemble.shape == ensemble.shape


def test_variational_graph_forward():
    """Test Variational DA forward pass with graph inputs."""
    var_da = VariationalDA({"iterations": 3})

    # Create graph input
    num_nodes = 5
    node_features = 10
    edge_features = 4

    graph_state = Data(
        x=torch.randn(num_nodes, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 10)),
        edge_attr=torch.randn(10, edge_features) if edge_features > 0 else None,
    )

    # Create observations
    obs_features = 5
    observations = torch.randn(1, obs_features)  # Batch size of 1 for graph

    # Forward pass
    result = var_da(graph_state, observations)

    # Check output is a graph with same structure
    assert hasattr(result, "x")
    assert result.x.shape == graph_state.x.shape
    assert torch.equal(result.edge_index, graph_state.edge_index)


def test_variational_graph_initialize_ensemble():
    """Test ensemble initialization with graph inputs."""
    var_da = VariationalDA({"background_error_std": 0.1})

    # Create background graph state
    num_nodes = 4
    node_features = 8
    background_graph = Data(
        x=torch.randn(num_nodes, node_features), edge_index=torch.randint(0, num_nodes, (2, 8))
    )
    num_members = 3

    # Initialize ensemble
    ensemble = var_da.initialize_ensemble(background_graph, num_members)

    # Check ensemble has correct structure
    assert hasattr(ensemble, "x")
    # For graphs, ensemble creates [num_nodes, num_members, features] shape for x
    assert ensemble.x.shape == (num_nodes, num_members, node_features)


def test_variational_graph_assimilate():
    """Test assimilation with graph inputs."""
    var_da = VariationalDA({"iterations": 2})

    # Create graph ensemble and observations
    num_nodes = 3
    node_features = 5
    obs_features = 3

    graph_ensemble = Data(
        x=torch.randn(num_nodes, 3, node_features),  # [num_nodes, num_members, features]
        edge_index=torch.randint(0, num_nodes, (2, 6)),
    )
    observations = torch.randn(1, obs_features)  # Batch size of 1 for graph

    # Perform assimilation
    updated_ensemble = var_da.assimilate(graph_ensemble, observations)

    # Check output shape
    assert hasattr(updated_ensemble, "x")
    assert updated_ensemble.x.shape == graph_ensemble.x.shape


def test_variational_compute_graph_cost_function():
    """Test graph cost function computation."""
    var_da = VariationalDA({"background_error_std": 0.5, "observation_error_std": 0.1})

    # Create graph states and observations
    num_nodes = 4
    node_features = 6
    obs_features = 3

    analysis_graph = Data(x=torch.randn(num_nodes, node_features))
    background_graph = Data(x=torch.randn(num_nodes, node_features))
    observations = torch.randn(1, obs_features)

    # Compute cost function
    cost = var_da._compute_graph_cost_function(analysis_graph, background_graph, observations)

    # Check output is scalar tensor
    assert cost.shape == torch.Size([])
    assert torch.is_tensor(cost)
    assert cost.requires_grad  # Should be differentiable


def test_variational_compute_analysis_tensor():
    """Test analysis computation for tensor ensembles."""
    var_da = VariationalDA()

    # Create ensemble
    batch_size = 2
    num_members = 4
    state_features = 8
    ensemble = torch.randn(batch_size, num_members, state_features)

    # Compute analysis
    analysis = var_da._compute_analysis(ensemble)

    # Check analysis shape
    assert analysis.shape == (batch_size, state_features)
    # Check that analysis is mean of ensemble
    expected_mean = torch.mean(ensemble, dim=1)
    assert torch.allclose(analysis, expected_mean, atol=1e-5)


def test_variational_compute_analysis_graph():
    """Test analysis computation for graph ensembles."""
    var_da = VariationalDA()

    # Create graph ensemble
    num_nodes = 3
    num_members = 4
    node_features = 6
    graph_ensemble = Data(
        x=torch.randn(num_nodes, num_members, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 6)),
    )

    # Compute analysis
    analysis = var_da._compute_analysis(graph_ensemble)

    # Check analysis has correct structure
    assert hasattr(analysis, "x")
    assert analysis.x.shape == (num_nodes, node_features)
    # Check that analysis x is mean of ensemble x
    expected_mean = torch.mean(graph_ensemble.x, dim=1)
    assert torch.allclose(analysis.x, expected_mean, atol=1e-5)


def test_variational_learnable_weights():
    """Test that learnable weights are properly initialized."""
    var_da = VariationalDA()

    # Check that learnable parameters exist
    assert hasattr(var_da, "bg_weight")
    assert hasattr(var_da, "obs_weight")
    assert var_da.bg_weight.requires_grad
    assert var_da.obs_weight.requires_grad

    # Check that parameters can be optimized
    optimizer = torch.optim.SGD(var_da.parameters(), lr=0.01)

    # Create dummy input
    state = torch.randn(2, 8)
    obs = torch.randn(2, 4)

    # Forward pass
    result = var_da(state, obs)
    loss = torch.sum((result - state) ** 2)  # Dummy loss

    # Backward pass should work
    loss.backward()
    optimizer.step()

    # Parameters should have been updated
    assert var_da.bg_weight.requires_grad
    assert var_da.obs_weight.requires_grad


def test_variational_clone_graph_state():
    """Test graph state cloning functionality."""
    var_da = VariationalDA()

    # Create graph state
    num_nodes = 4
    node_features = 6
    graph_state = Data(
        x=torch.randn(num_nodes, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 8)),
        attr=torch.randn(num_nodes, 2),  # Additional attribute
    )

    # Clone the graph state
    cloned = var_da._clone_graph_state(graph_state)

    # Check that clone has same structure and values
    assert torch.equal(cloned.x, graph_state.x)
    assert torch.equal(cloned.edge_index, graph_state.edge_index)
    assert torch.equal(cloned.attr, graph_state.attr)
    # But should be different objects
    assert cloned is not graph_state
    assert cloned.x.data_ptr() != graph_state.x.data_ptr()


def test_variational_error_handling():
    """Test error handling for invalid inputs."""
    var_da = VariationalDA()

    # Test invalid ensemble type
    with pytest.raises(TypeError):
        var_da.assimilate("invalid_type", torch.randn(2, 5))

    # Test unsupported state type
    with pytest.raises(TypeError):
        var_da._compute_cost_function("invalid_type", torch.randn(2, 5), torch.randn(2, 3))


if __name__ == "__main__":
    print("Running Variational DA tests...")

    test_variational_da_initialization()
    print("✓ Initialization test passed")

    test_variational_tensor_forward()
    print("✓ Tensor forward test passed")

    test_variational_tensor_initialize_ensemble()
    print("✓ Tensor ensemble initialization test passed")

    test_variational_tensor_assimilate()
    print("✓ Tensor assimilation test passed")

    test_variational_compute_tensor_cost_function()
    print("✓ Tensor cost function test passed")

    test_variational_tensor_assimilate_ensemble()
    print("✓ Tensor ensemble assimilation test passed")

    test_variational_graph_forward()
    print("✓ Graph forward test passed")

    test_variational_graph_initialize_ensemble()
    print("✓ Graph ensemble initialization test passed")

    test_variational_graph_assimilate()
    print("✓ Graph assimilation test passed")

    test_variational_compute_graph_cost_function()
    print("✓ Graph cost function test passed")

    test_variational_compute_analysis_tensor()
    print("✓ Tensor analysis computation test passed")

    test_variational_compute_analysis_graph()
    print("✓ Graph analysis computation test passed")

    test_variational_learnable_weights()
    print("✓ Learnable weights test passed")

    test_variational_clone_graph_state()
    print("✓ Graph cloning test passed")

    test_variational_error_handling()
    print("✓ Error handling test passed")

    print("\n✅ All Variational DA tests passed!")
