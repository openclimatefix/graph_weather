import pytest
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, '../../../graph_weather/models/data_assimilation')

# Execute modules directly to avoid import issues
exec(open('graph_weather/models/data_assimilation/data_assimilation_base.py').read())


class MockDA(DataAssimilationBase):
    """Mock implementation of DataAssimilationBase for testing purposes."""

    def initialize_ensemble(self, background_state, num_members):
        return self.ensemble_generator.generate_ensemble(background_state, num_members)

    def assimilate(self, ensemble, observations):
        return ensemble  # Return unchanged for testing

    def _compute_analysis(self, ensemble):
        if isinstance(ensemble, torch.Tensor):
            return torch.mean(ensemble, dim=1)
        elif isinstance(ensemble, Data):
            return ensemble  # Return as is for testing
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")


def test_ensemble_generator_tensor():
    """Test ensemble generation for tensor inputs."""
    generator = EnsembleGenerator(noise_std=0.1, method="gaussian")
    
    # Test tensor input
    state = torch.randn(2, 5, 3)  # [batch, nodes, features]
    ensemble = generator.generate_ensemble(state, 4)
    
    assert ensemble.shape == (2, 4, 5, 3)  # [batch, members, nodes, features]
    assert not torch.equal(state, ensemble[:, 0])  # Should have noise added


def test_ensemble_generator_graph():
    """Test ensemble generation for graph inputs."""
    generator = EnsembleGenerator(noise_std=0.1, method="gaussian")
    
    # Test graph input
    x = torch.randn(10, 4)  # Node features
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    graph_state = Data(x=x, edge_index=edge_index)
    
    ensemble = generator.generate_ensemble(graph_state, 3)
    
    # Check that ensemble preserves structure
    assert hasattr(ensemble, 'x')
    assert hasattr(ensemble, 'edge_index')
    assert ensemble.x.shape[1] == 3  # Ensemble dimension


def test_data_assimilation_base_abstract_methods():
    """Test that abstract methods are properly defined."""
    config = {"param": "value"}
    da_module = MockDA(config)
    
    assert da_module.config == config
    
    # Test ensemble generation
    state = torch.randn(2, 5, 3)
    ensemble = da_module.initialize_ensemble(state, 4)
    assert ensemble.shape == (2, 4, 5, 3)


def test_compute_analysis_tensor():
    """Test analysis computation for tensor ensembles."""
    da_module = MockDA({})
    
    # Create ensemble: [batch, members, nodes, features]
    ensemble = torch.stack([
        torch.ones(2, 5, 3),      # First member
        2 * torch.ones(2, 5, 3),  # Second member
        3 * torch.ones(2, 5, 3),  # Third member
    ], dim=1)  # Shape: [2, 3, 5, 3]
    
    analysis = da_module._compute_analysis(ensemble)
    
    # Mean should be (1 + 2 + 3) / 3 = 2
    expected = 2 * torch.ones(2, 5, 3)
    assert torch.allclose(analysis, expected)


if __name__ == "__main__":
    test_ensemble_generator_tensor()
    test_ensemble_generator_graph()
    test_data_assimilation_base_abstract_methods()
    test_compute_analysis_tensor()
    print("All tests passed!")