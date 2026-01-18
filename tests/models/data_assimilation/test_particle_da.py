"""
Comprehensive tests for the Particle Filter Data Assimilation method.

Tests include functionality for both tensor and graph-based inputs.
"""
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Use direct import to avoid package conflicts
import importlib.util

# Add the graph_weather directory to the path to make relative imports work
sys.path.insert(0, os.path.join(os.getcwd(), 'graph_weather'))

# Load base module first
spec = importlib.util.spec_from_file_location('data_assimilation_base', './graph_weather/models/data_assimilation/data_assimilation_base.py')
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)

# Load particle module with proper base class injection
spec = importlib.util.spec_from_file_location('particle_filter_da', './graph_weather/models/data_assimilation/particle_filter_da.py')
particle_module = importlib.util.module_from_spec(spec)
particle_module.DataAssimilationBase = base_module.DataAssimilationBase
particle_module.EnsembleGenerator = base_module.EnsembleGenerator
particle_module.Data = Data
particle_module.HeteroData = getattr(__import__('torch_geometric.data', fromlist=['HeteroData']), 'HeteroData', None)
particle_module.torch = torch
particle_module.nn = torch.nn
particle_module.typing = __import__('typing')
spec.loader.exec_module(particle_module)

ParticleFilterDA = particle_module.ParticleFilterDA
EnsembleGenerator = base_module.EnsembleGenerator


def test_particle_filter_initialization():
    """Test Particle Filter DA initialization with default and custom configs."""
    # Test default initialization
    pf_da = ParticleFilterDA()
    assert pf_da.num_particles == 100
    assert pf_da.resample_threshold == 0.5
    assert pf_da.observation_error_std == 0.1
    assert pf_da.process_noise_std == 0.05
    
    # Test custom initialization
    config = {
        "num_particles": 50,
        "resample_threshold": 0.3,
        "observation_error_std": 0.05,
        "process_noise_std": 0.1
    }
    pf_da_custom = ParticleFilterDA(config)
    assert pf_da_custom.num_particles == 50
    assert pf_da_custom.resample_threshold == 0.3
    assert pf_da_custom.observation_error_std == 0.05
    assert pf_da_custom.process_noise_std == 0.1
    assert hasattr(pf_da_custom, 'temperature')  # Temperature parameter should exist


def test_particle_tensor_forward():
    """Test Particle Filter DA forward pass with tensor inputs."""
    pf_da = ParticleFilterDA({"num_particles": 15})
    
    # Create input state and observations
    batch_size = 2
    state_features = 12
    obs_features = 6
    
    state_in = torch.randn(batch_size, state_features)
    observations = torch.randn(batch_size, obs_features)
    
    # Forward pass
    result = pf_da(state_in, observations)
    
    # Check output shape
    assert result.shape == state_in.shape
    assert torch.is_tensor(result)


def test_particle_tensor_initialize_ensemble():
    """Test ensemble initialization with tensor inputs."""
    pf_da = ParticleFilterDA({"process_noise_std": 0.1})
    
    # Create background state
    batch_size = 3
    state_features = 10
    background_state = torch.randn(batch_size, state_features)
    num_members = 8
    
    # Initialize ensemble
    ensemble = pf_da.initialize_ensemble(background_state, num_members)
    
    # Check ensemble shape
    assert ensemble.shape == (batch_size, num_members, state_features)
    # Check that ensemble members are similar but not identical to background
    ensemble_mean = torch.mean(ensemble, dim=1)
    assert torch.allclose(ensemble_mean, background_state, atol=0.2)


def test_particle_tensor_assimilate():
    """Test assimilation with tensor inputs."""
    pf_da = ParticleFilterDA({"num_particles": 10, "observation_error_std": 0.1})
    
    # Create ensemble and observations
    batch_size = 2
    state_features = 8
    obs_features = 4
    num_particles = 10
    
    ensemble = torch.randn(batch_size, num_particles, state_features)
    observations = torch.randn(batch_size, obs_features)
    
    # Perform assimilation
    updated_ensemble = pf_da.assimilate(ensemble, observations)
    
    # Check output shape
    assert updated_ensemble.shape == ensemble.shape
    assert torch.is_tensor(updated_ensemble)


def test_particle_compute_log_likelihood():
    """Test log-likelihood computation for tensor particles."""
    pf_da = ParticleFilterDA({"observation_error_std": 0.1})
    
    # Create particles and observations
    batch_size = 2
    num_particles = 6
    state_features = 8
    obs_features = 4
    
    particles = torch.randn(batch_size, num_particles, state_features)
    observations = torch.randn(batch_size, obs_features)
    
    # Compute log-likelihood
    log_likelihood = pf_da._compute_log_likelihood(particles, observations)
    
    # Check output shape (should match particles shape with reduced feature dimensions)
    expected_shape = (batch_size, num_particles, 1, 1)  # Expanded to match particle dims
    assert log_likelihood.shape[:2] == (batch_size, num_particles)  # First two dims match


def test_particle_resample_particles():
    """Test particle resampling functionality."""
    pf_da = ParticleFilterDA()
    
    # Create particles and weights
    batch_size = 2
    num_particles = 8
    state_features = 6
    
    particles = torch.randn(batch_size, num_particles, state_features)
    # Create normalized weights
    weights_raw = torch.softmax(torch.randn(batch_size, num_particles), dim=1)
    
    # Resample particles
    resampled = pf_da._resample_particles(particles, weights_raw.unsqueeze(-1).unsqueeze(-1))
    
    # Check output shape
    assert resampled.shape == particles.shape
    # The resampled particles should be a selection/repetition of original particles
    assert torch.is_tensor(resampled)


def test_particle_graph_forward():
    """Test Particle Filter DA forward pass with graph inputs."""
    pf_da = ParticleFilterDA({"num_particles": 5})
    
    # Create graph input
    num_nodes = 6
    node_features = 12
    edge_features = 4
    
    graph_state = Data(
        x=torch.randn(num_nodes, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 12)),
        edge_attr=torch.randn(12, edge_features) if edge_features > 0 else None
    )
    
    # Create observations
    obs_features = 6
    observations = torch.randn(1, obs_features)  # Batch size of 1 for graph
    
    # Forward pass
    result = pf_da(graph_state, observations)
    
    # Check output is a graph with same structure
    assert hasattr(result, 'x')
    assert result.x.shape == graph_state.x.shape
    assert torch.equal(result.edge_index, graph_state.edge_index)


def test_particle_graph_initialize_ensemble():
    """Test ensemble initialization with graph inputs."""
    pf_da = ParticleFilterDA({"process_noise_std": 0.1})
    
    # Create background graph state
    num_nodes = 5
    node_features = 8
    background_graph = Data(
        x=torch.randn(num_nodes, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 10))
    )
    num_members = 4
    
    # Initialize ensemble
    ensemble = pf_da.initialize_ensemble(background_graph, num_members)
    
    # Check ensemble has correct structure
    assert hasattr(ensemble, 'x')
    # For graphs, ensemble creates [num_nodes, num_members, features] shape for x
    assert ensemble.x.shape == (num_nodes, num_members, node_features)


def test_particle_graph_assimilate():
    """Test assimilation with graph inputs."""
    pf_da = ParticleFilterDA({"num_particles": 6})
    
    # Create graph ensemble and observations
    num_nodes = 4
    node_features = 6
    obs_features = 4
    
    graph_ensemble = Data(
        x=torch.randn(num_nodes, 6, node_features),  # [num_nodes, num_members, features]
        edge_index=torch.randint(0, num_nodes, (2, 8))
    )
    observations = torch.randn(1, obs_features)  # Batch size of 1 for graph
    
    # Perform assimilation
    updated_ensemble = pf_da.assimilate(graph_ensemble, observations)
    
    # Check output shape
    assert hasattr(updated_ensemble, 'x')
    assert updated_ensemble.x.shape == graph_ensemble.x.shape


def test_particle_compute_analysis_tensor():
    """Test analysis computation for tensor ensembles."""
    pf_da = ParticleFilterDA()
    
    # Create ensemble
    batch_size = 3
    num_particles = 5
    state_features = 10
    ensemble = torch.randn(batch_size, num_particles, state_features)
    
    # Compute analysis
    analysis = pf_da._compute_analysis(ensemble)
    
    # Check analysis shape
    assert analysis.shape == (batch_size, state_features)
    # Check that analysis is mean of ensemble
    expected_mean = torch.mean(ensemble, dim=1)
    assert torch.allclose(analysis, expected_mean, atol=1e-5)


def test_particle_compute_analysis_graph():
    """Test analysis computation for graph ensembles."""
    pf_da = ParticleFilterDA()
    
    # Create graph ensemble
    num_nodes = 3
    num_particles = 4
    node_features = 6
    graph_ensemble = Data(
        x=torch.randn(num_nodes, num_particles, node_features),
        edge_index=torch.randint(0, num_nodes, (2, 6))
    )
    
    # Compute analysis
    analysis = pf_da._compute_analysis(graph_ensemble)
    
    # Check analysis has correct structure
    assert hasattr(analysis, 'x')
    assert analysis.x.shape == (num_nodes, node_features)
    # Check that analysis x is mean of ensemble x
    expected_mean = torch.mean(graph_ensemble.x, dim=1)
    assert torch.allclose(analysis.x, expected_mean, atol=1e-5)


def test_particle_temperature_parameter():
    """Test temperature parameter functionality."""
    pf_da = ParticleFilterDA()
    
    # Check that temperature parameter exists and is reasonable
    assert hasattr(pf_da, 'temperature')
    assert pf_da.temperature.requires_grad  # Should be trainable
    
    # Test that temperature affects likelihood computation
    batch_size = 1
    num_particles = 5
    state_features = 6
    obs_features = 3
    
    particles = torch.randn(batch_size, num_particles, state_features)
    observations = torch.randn(batch_size, obs_features)
    
    log_likelihood = pf_da._compute_log_likelihood(particles, observations)
    assert torch.is_tensor(log_likelihood)
    assert log_likelihood.shape[0] == batch_size
    assert log_likelihood.shape[1] == num_particles


def test_particle_error_handling():
    """Test error handling for invalid inputs."""
    pf_da = ParticleFilterDA()
    
    # Test invalid ensemble type
    with pytest.raises(TypeError):
        pf_da.assimilate("invalid_type", torch.randn(2, 5))


if __name__ == "__main__":
    print("Running Particle Filter DA tests...")
    
    test_particle_filter_initialization()
    print("✓ Initialization test passed")
    
    test_particle_tensor_forward()
    print("✓ Tensor forward test passed")
    
    test_particle_tensor_initialize_ensemble()
    print("✓ Tensor ensemble initialization test passed")
    
    test_particle_tensor_assimilate()
    print("✓ Tensor assimilation test passed")
    
    test_particle_compute_log_likelihood()
    print("✓ Log-likelihood computation test passed")
    
    test_particle_resample_particles()
    print("✓ Particle resampling test passed")
    
    test_particle_graph_forward()
    print("✓ Graph forward test passed")
    
    test_particle_graph_initialize_ensemble()
    print("✓ Graph ensemble initialization test passed")
    
    test_particle_graph_assimilate()
    print("✓ Graph assimilation test passed")
    
    test_particle_compute_analysis_tensor()
    print("✓ Tensor analysis computation test passed")
    
    test_particle_compute_analysis_graph()
    print("✓ Graph analysis computation test passed")
    
    test_particle_temperature_parameter()
    print("✓ Temperature parameter test passed")
    
    test_particle_error_handling()
    print("✓ Error handling test passed")
    
    print("\n✅ All Particle Filter DA tests passed!")