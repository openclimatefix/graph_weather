"""Base classes for data assimilation modules."""
import abc
from typing import Union, Dict, Any, Optional
import torch
from torch_geometric.data import Data


class EnsembleGenerator:
    """Class to generate ensemble members from a background state."""
    
    def __init__(self, noise_std: float = 0.1, method: str = "gaussian"):
        self.noise_std = noise_std
        self.method = method
        
    def generate_ensemble(self, state: Union[torch.Tensor, Data], num_members: int):
        if isinstance(state, torch.Tensor):
            return self._generate_tensor_ensemble(state, num_members)
        elif isinstance(state, Data):
            return self._generate_graph_ensemble(state, num_members)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
            
    def _generate_tensor_ensemble(self, state: torch.Tensor, num_members: int) -> torch.Tensor:
        batch_size, nodes, features = state.shape
        ensemble = torch.zeros(batch_size, num_members, nodes, features, device=state.device)
        
        for i in range(num_members):
            if self.method == "gaussian":
                noise = torch.randn_like(state) * self.noise_std
                ensemble[:, i] = state + noise
            elif self.method == "dropout":
                mask = torch.bernoulli(torch.ones_like(state) * 0.9)  # Keep 90% of values
                noise = torch.randn_like(state) * self.noise_std * 0.1
                ensemble[:, i] = (state * mask) + noise
            elif self.method == "perturbation":
                perturbation = torch.randn_like(state) * self.noise_std * torch.linspace(0.1, 1.0, num_members)[i]
                ensemble[:, i] = state + perturbation
            else:
                raise ValueError(f"Unknown ensemble generation method: {self.method}")
                
        return ensemble
    
    def _generate_graph_ensemble(self, state: Data, num_members: int) -> Data:
        x_expanded = torch.zeros(state.x.shape[0], num_members, state.x.shape[1], device=state.x.device)
        
        for i in range(num_members):
            if self.method == "gaussian":
                noise = torch.randn_like(state.x) * self.noise_std
                x_expanded[:, i] = state.x + noise
            elif self.method == "dropout":
                mask = torch.bernoulli(torch.ones_like(state.x) * 0.9)
                noise = torch.randn_like(state.x) * self.noise_std * 0.1
                x_expanded[:, i] = (state.x * mask) + noise
            elif self.method == "perturbation":
                perturbation = torch.randn_like(state.x) * self.noise_std * torch.linspace(0.1, 1.0, num_members)[i]
                x_expanded[:, i] = state.x + perturbation
            else:
                raise ValueError(f"Unknown ensemble generation method: {self.method}")
        
        new_state = Data(
            x=x_expanded,
            edge_index=state.edge_index,
            edge_attr=getattr(state, 'edge_attr', None),
            pos=getattr(state, 'pos', None)
        )
        
        return new_state


class DataAssimilationBase(abc.ABC):
    """Abstract base class for data assimilation modules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensemble_generator = EnsembleGenerator(
            noise_std=config.get('noise_std', 0.1),
            method=config.get('ensemble_method', 'gaussian')
        )
    
    @abc.abstractmethod
    def initialize_ensemble(self, background_state: Union[torch.Tensor, Data], num_members: int):
        pass
    
    @abc.abstractmethod
    def assimilate(self, ensemble: Union[torch.Tensor, Data], observations: torch.Tensor):
        pass
    
    @abc.abstractmethod
    def _compute_analysis(self, ensemble: Union[torch.Tensor, Data]) -> Union[torch.Tensor, Data]:
        pass
    
    def forward(self, state: Union[torch.Tensor, Data], observations: torch.Tensor, num_ensemble: int = 10):
        ensemble = self.initialize_ensemble(state, num_ensemble)
        updated_ensemble = self.assimilate(ensemble, observations)
        analysis = self._compute_analysis(updated_ensemble)
        
        return updated_ensemble, analysis