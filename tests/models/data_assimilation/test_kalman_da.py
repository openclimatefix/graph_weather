"""Tests for Kalman Filter Data Assimilation."""

import importlib.util
import os

import pytest
import torch
from torch_geometric.data import Data


def _load_kalman_class():
    """Load KalmanFilterDA directly from file path."""
    kf_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            os.pardir,
            "graph_weather",
            "models",
            "data_assimilation",
            "kalman_filter_da.py",
        )
    )
    spec = importlib.util.spec_from_file_location("kalman_filter_da", kf_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KalmanFilterDA


KalmanFilterDA = _load_kalman_class()


def test_kalman_filter_initialization():
    """Test initialization with default and custom configs."""
    kf = KalmanFilterDA()
    assert kf.ensemble_size == 20
    assert kf.inflation_factor == 1.1
    assert kf.observation_error_std == 0.1
    assert kf.background_error_std == 0.5

    custom = KalmanFilterDA(
        {
            "ensemble_size": 30,
            "inflation_factor": 1.2,
            "observation_error_std": 0.05,
            "background_error_std": 0.3,
            "adaptive_inflation": False,
        }
    )
    assert custom.ensemble_size == 30
    assert custom.inflation_factor == 1.2
    assert custom.observation_error_std == 0.05
    assert custom.background_error_std == 0.3
    assert not custom.adaptive_inflation


def test_kalman_tensor_forward():
    """Test forward pass with tensor inputs."""
    kf = KalmanFilterDA({"ensemble_size": 10})
    state_in = torch.randn(2, 16)
    observations = torch.randn(2, 8)

    result = kf(state_in, observations)

    assert result.shape == state_in.shape
    assert torch.is_tensor(result)


def test_kalman_tensor_initialize_ensemble():
    """Test ensemble initialization with tensor inputs."""
    kf = KalmanFilterDA({"background_error_std": 0.1})
    bg = torch.randn(3, 12)

    ensemble = kf.initialize_ensemble(bg, 5)

    assert ensemble.shape == (3, 5, 12)
    assert torch.allclose(ensemble.mean(dim=1), bg, atol=0.2)


def test_kalman_tensor_assimilate():
    """Test assimilation with tensor inputs."""
    kf = KalmanFilterDA({"ensemble_size": 8, "observation_error_std": 0.1})
    ensemble = torch.randn(2, 8, 10)
    obs = torch.randn(2, 5)

    updated = kf.assimilate(ensemble, obs)

    assert updated.shape == ensemble.shape
    assert torch.is_tensor(updated)


def test_kalman_graph_forward():
    """Test forward pass with graph inputs."""
    kf = KalmanFilterDA({"ensemble_size": 5})
    graph = Data(
        x=torch.randn(8, 16),
        edge_index=torch.randint(0, 8, (2, 15)),
        edge_attr=torch.randn(15, 4),
    )
    obs = torch.randn(1, 8)

    result = kf(graph, obs)

    assert hasattr(result, "x")
    assert result.x.shape == graph.x.shape
    assert torch.equal(result.edge_index, graph.edge_index)


def test_kalman_graph_initialize_ensemble():
    """Test ensemble initialization with graph inputs."""
    kf = KalmanFilterDA({"background_error_std": 0.1})
    graph = Data(
        x=torch.randn(6, 10),
        edge_index=torch.randint(0, 6, (2, 10)),
    )

    ensemble = kf.initialize_ensemble(graph, 4)

    assert hasattr(ensemble, "x")
    assert ensemble.x.shape == (6, 4, 10)


def test_kalman_graph_assimilate():
    """Test assimilation with graph inputs."""
    kf = KalmanFilterDA({"ensemble_size": 6})
    graph = Data(
        x=torch.randn(5, 6, 8),
        edge_index=torch.randint(0, 5, (2, 10)),
    )
    obs = torch.randn(1, 6)

    updated = kf.assimilate(graph, obs)

    assert hasattr(updated, "x")
    assert updated.x.shape == graph.x.shape


def test_kalman_compute_analysis_tensor():
    """Test analysis computation for tensor ensembles."""
    kf = KalmanFilterDA()
    ensemble = torch.randn(3, 5, 12)

    analysis = kf._compute_analysis(ensemble)

    assert analysis.shape == (3, 12)
    assert torch.allclose(analysis, ensemble.mean(dim=1), atol=1e-5)


def test_kalman_compute_analysis_graph():
    """Test analysis computation for graph ensembles."""
    kf = KalmanFilterDA()
    graph = Data(
        x=torch.randn(4, 3, 8),
        edge_index=torch.randint(0, 4, (2, 8)),
    )

    analysis = kf._compute_analysis(graph)

    assert hasattr(analysis, "x")
    assert analysis.x.shape == (4, 8)
    assert torch.allclose(analysis.x, graph.x.mean(dim=1), atol=1e-5)


def test_kalman_different_inflation_modes():
    """Test adaptive vs fixed inflation."""
    adaptive = KalmanFilterDA({"adaptive_inflation": True})
    assert adaptive.adaptive_inflation
    assert hasattr(adaptive, "inflation_param")

    fixed = KalmanFilterDA({"adaptive_inflation": False, "inflation_factor": 1.2})
    assert not fixed.adaptive_inflation
    assert not hasattr(fixed, "inflation_param")


def test_kalman_error_handling():
    """Test error handling for invalid inputs."""
    kf = KalmanFilterDA()
    with pytest.raises(TypeError):
        kf.assimilate("invalid_type", torch.randn(2, 5))
