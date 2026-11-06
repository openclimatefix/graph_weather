"""
Comprehensive tests for the Data Assimilation Interface.

Tests include functionality for strategy switching and unified interface.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Literal
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

# Load all DA modules with proper base class injection
modules_to_load = ["kalman_filter_da", "particle_filter_da", "variational_da"]

loaded_modules = {}
for module_name in modules_to_load:
    spec = importlib.util.spec_from_file_location(
        module_name, f"./graph_weather/models/data_assimilation/{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.DataAssimilationBase = base_module.DataAssimilationBase
    module.EnsembleGenerator = base_module.EnsembleGenerator
    module.Data = Data
    module.HeteroData = getattr(
        __import__("torch_geometric.data", fromlist=["HeteroData"]), "HeteroData", None
    )
    module.torch = torch
    module.nn = torch.nn
    module.F = torch.nn.functional
    module.typing = __import__("typing")
    module.abc = __import__("abc")
    spec.loader.exec_module(module)
    loaded_modules[module_name] = module

# Load interface module with all dependencies
spec = importlib.util.spec_from_file_location(
    "interface", "./graph_weather/models/data_assimilation/interface.py"
)
interface_module = importlib.util.module_from_spec(spec)
interface_module.KalmanFilterDA = loaded_modules["kalman_filter_da"].KalmanFilterDA
interface_module.ParticleFilterDA = loaded_modules["particle_filter_da"].ParticleFilterDA
interface_module.VariationalDA = loaded_modules["variational_da"].VariationalDA
interface_module.Data = Data
interface_module.torch = torch
interface_module.nn = torch.nn
interface_module.Literal = Literal
interface_module.Dict = getattr(__import__("typing"), "Dict", dict)
interface_module.Any = getattr(__import__("typing"), "Any", object)
interface_module.Optional = getattr(__import__("typing"), "Optional", type(None))
spec.loader.exec_module(interface_module)

DAInterface = interface_module.DAInterface


def test_da_interface_initialization():
    """Test DA Interface initialization with different strategies."""
    # Test Kalman filter initialization
    da_kf = DAInterface(strategy="kalman")
    assert da_kf.strategy == "kalman"
    assert da_kf.get_strategy() == "kalman"

    # Test Particle filter initialization
    da_pf = DAInterface(strategy="particle")
    assert da_pf.strategy == "particle"
    assert da_pf.get_strategy() == "particle"

    # Test Variational DA initialization
    da_var = DAInterface(strategy="variational")
    assert da_var.strategy == "variational"
    assert da_var.get_strategy() == "variational"

    # Test with custom config
    config = {"ensemble_size": 15, "observation_error_std": 0.05}
    da_kf_config = DAInterface(strategy="kalman", config=config)
    assert da_kf_config.strategy == "kalman"


def test_da_interface_forward():
    """Test DA Interface forward pass with different strategies."""
    # Test Kalman filter
    da_kf = DAInterface(strategy="kalman", config={"ensemble_size": 8})
    state = torch.randn(2, 10)
    obs = torch.randn(2, 5)
    result = da_kf(state, obs)
    assert result.shape == state.shape

    # Test Particle filter
    da_pf = DAInterface(strategy="particle", config={"num_particles": 8})
    result = da_pf(state, obs)
    assert result.shape == state.shape

    # Test Variational DA
    da_var = DAInterface(strategy="variational", config={"iterations": 3})
    result = da_var(state, obs)
    assert result.shape == state.shape


def test_da_interface_ensemble_operations():
    """Test DA Interface ensemble operations with different strategies."""
    strategies = ["kalman", "particle", "variational"]

    for strategy in strategies:
        if strategy == "kalman":
            da = DAInterface(strategy=strategy, config={"ensemble_size": 6})
            num_members = 6
        elif strategy == "particle":
            da = DAInterface(strategy=strategy, config={"num_particles": 6})
            num_members = 6
        else:  # variational
            da = DAInterface(strategy=strategy, config={"iterations": 2})
            num_members = 4  # Using smaller number for variational for efficiency

        # Test ensemble initialization
        background_state = torch.randn(2, 8)
        ensemble = da.initialize_ensemble(background_state, num_members)
        assert ensemble.shape == (2, num_members, 8)

        # Test assimilation
        observations = torch.randn(2, 4)
        updated_ensemble = da.assimilate(ensemble, observations)
        assert updated_ensemble.shape == ensemble.shape


def test_da_interface_strategy_switching():
    """Test switching between DA strategies."""
    da_interface = DAInterface(strategy="kalman", config={"ensemble_size": 10})

    # Initial state
    state = torch.randn(2, 8)
    obs = torch.randn(2, 4)

    # Get initial result
    result_before = da_interface(state, obs)
    assert da_interface.get_strategy() == "kalman"

    # Switch to particle filter
    da_interface.switch_strategy("particle", {"num_particles": 8})
    assert da_interface.get_strategy() == "particle"

    # Get result after switch
    result_after = da_interface(state, obs)
    assert result_after.shape == state.shape

    # Switch to variational
    da_interface.switch_strategy("variational", {"iterations": 3})
    assert da_interface.get_strategy() == "variational"

    # Get result after second switch
    result_final = da_interface(state, obs)
    assert result_final.shape == state.shape


def test_create_da_module_function():
    """Test the create_da_module factory function."""
    # Create Kalman filter
    da_kf = create_da_module("kalman", {"ensemble_size": 12})
    assert da_kf.__class__.__name__ == "KalmanFilterDA"

    # Create Particle filter
    da_pf = create_da_module("particle", {"num_particles": 15})
    assert da_pf.__class__.__name__ == "ParticleFilterDA"

    # Create Variational DA
    da_var = create_da_module("variational", {"iterations": 5})
    assert da_var.__class__.__name__ == "VariationalDA"

    # Test error for unknown strategy
    try:
        create_da_module("unknown")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_model_integrated_da_basic():
    """Test basic functionality of ModelIntegratedDA."""

    # Create a simple dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x  # Identity function for testing

    # Create DA interface
    da_interface = DAInterface(strategy="kalman", config={"ensemble_size": 8})

    # Create integrated DA model
    integrated_da = ModelIntegratedDA(
        base_model=DummyModel(), da_interface=da_interface, ensemble_size=8, enable_da=True
    )

    # Test forward pass without observations
    inputs = torch.randn(2, 10)
    result = integrated_da(inputs, observations=None)
    assert result.shape == inputs.shape

    # Test forward pass with observations
    obs = torch.randn(2, 5)
    result = integrated_da(inputs, observations=obs)
    assert result.shape == inputs.shape

    # Test with return_ensemble=True
    result = integrated_da(inputs, observations=obs, return_ensemble=True)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "ensemble" in result
    assert result["prediction"].shape == inputs.shape


def test_model_integrated_da_disable_da():
    """Test disabling DA in ModelIntegratedDA."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x  # Identity function for testing

    da_interface = DAInterface(strategy="kalman", config={"ensemble_size": 6})

    integrated_da = ModelIntegratedDA(
        base_model=DummyModel(),
        da_interface=da_interface,
        ensemble_size=6,
        enable_da=False,  # DA disabled
    )

    inputs = torch.randn(2, 8)
    obs = torch.randn(2, 4)

    # When DA is disabled, should return base model output without DA
    result = integrated_da(inputs, observations=obs)
    assert result.shape == inputs.shape


def test_integrate_da_with_model():
    """Test the integrate_da_with_model convenience function."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    # Test with Kalman filter
    integrated_model_kf = integrate_da_with_model(
        DummyModel(), da_strategy="kalman", da_config={"ensemble_size": 5}, ensemble_size=5
    )
    assert isinstance(integrated_model_kf, ModelIntegratedDA)

    # Test with Particle filter
    integrated_model_pf = integrate_da_with_model(
        DummyModel(), da_strategy="particle", da_config={"num_particles": 6}, ensemble_size=6
    )
    assert isinstance(integrated_model_pf, ModelIntegratedDA)

    # Test with Variational DA
    integrated_model_var = integrate_da_with_model(
        DummyModel(), da_strategy="variational", da_config={"iterations": 3}, ensemble_size=4
    )
    assert isinstance(integrated_model_var, ModelIntegratedDA)


def test_da_interface_graph_operations():
    """Test DA Interface with graph inputs."""
    strategies = ["kalman", "particle", "variational"]

    for strategy in strategies:
        if strategy == "kalman":
            da = DAInterface(strategy=strategy, config={"ensemble_size": 4})
            num_members = 4
        elif strategy == "particle":
            da = DAInterface(strategy=strategy, config={"num_particles": 4})
            num_members = 4
        else:  # variational
            da = DAInterface(strategy=strategy, config={"iterations": 2})
            num_members = 3

        # Create graph input
        num_nodes = 5
        node_features = 8
        graph_state = Data(
            x=torch.randn(num_nodes, node_features), edge_index=torch.randint(0, num_nodes, (2, 10))
        )

        # Create observations
        obs_features = 4
        observations = torch.randn(1, obs_features)

        # Test forward pass
        result = da(graph_state, observations)
        assert hasattr(result, "x")
        assert result.x.shape == graph_state.x.shape

        # Test ensemble initialization
        ensemble = da.initialize_ensemble(graph_state, num_members)
        assert hasattr(ensemble, "x")
        assert ensemble.x.shape == (num_nodes, num_members, node_features)

        # Test assimilation
        updated_ensemble = da.assimilate(ensemble, observations)
        assert updated_ensemble.x.shape == ensemble.x.shape


def test_da_interface_error_handling():
    """Test error handling in DA Interface."""
    # Test invalid strategy initialization
    try:
        DAInterface(strategy="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    # Test invalid strategy switching
    da_interface = DAInterface(strategy="kalman")
    try:
        da_interface.switch_strategy("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_model_integration_end_to_end():
    """Test end-to-end integration of DA with a model."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    # Test with different DA strategies
    strategies_configs = [
        ("kalman", {"ensemble_size": 6}),
        ("particle", {"num_particles": 6}),
        ("variational", {"iterations": 3}),
    ]

    for strategy, config in strategies_configs:
        # Create integrated model
        integrated_model = integrate_da_with_model(
            SimpleModel(),
            da_strategy=strategy,
            da_config=config,
            ensemble_size=6 if strategy != "variational" else 4,
        )

        # Test inputs
        inputs = torch.randn(3, 10)
        observations = torch.randn(3, 8)

        # Forward pass with DA
        result = integrated_model(inputs, observations=observations)
        assert result.shape == inputs.shape

        # Forward pass without DA
        result_no_da = integrated_model(inputs, observations=None)
        assert result_no_da.shape == inputs.shape

        # Forward pass with ensemble return
        result_with_ensemble = integrated_model(
            inputs, observations=observations, return_ensemble=True
        )
        assert isinstance(result_with_ensemble, dict)
        assert "prediction" in result_with_ensemble
        assert "ensemble" in result_with_ensemble


if __name__ == "__main__":
    print("Running DA Interface tests...")

    test_da_interface_initialization()
    print("✓ Interface initialization test passed")

    test_da_interface_forward()
    print("✓ Interface forward test passed")

    test_da_interface_ensemble_operations()
    print("✓ Interface ensemble operations test passed")

    test_da_interface_strategy_switching()
    print("✓ Strategy switching test passed")

    test_create_da_module_function()
    print("✓ Factory function test passed")

    test_model_integrated_da_basic()
    print("✓ Integrated DA basic test passed")

    test_model_integrated_da_disable_da()
    print("✓ Integrated DA disable test passed")

    test_integrate_da_with_model()
    print("✓ Integration function test passed")

    test_da_interface_graph_operations()
    print("✓ Graph operations test passed")

    test_da_interface_error_handling()
    print("✓ Error handling test passed")

    test_model_integration_end_to_end()
    print("✓ End-to-end integration test passed")

    print("\n✅ All DA Interface tests passed!")
