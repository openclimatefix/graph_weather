import abc
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData


class DataAssimilationBase(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__()
        self.config = config or {}

    @abc.abstractmethod
    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Data, HeteroData]:

        pass

    @abc.abstractmethod
    def initialize_ensemble(
        self, background_state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:

        pass

    @abc.abstractmethod
    def assimilate(
        self, ensemble: Union[torch.Tensor, Data, HeteroData], observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:

        pass


class EnsembleGenerator(nn.Module):

    def __init__(self, noise_std: float = 0.1, method: str = "gaussian"):

        super().__init__()
        self.noise_std = noise_std
        self.method = method

    def forward(
        self, state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if isinstance(state, torch.Tensor):
            return self._generate_tensor_ensemble(state, num_members)
        elif isinstance(state, (Data, HeteroData)):
            return self._generate_graph_ensemble(state, num_members)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

    def _generate_tensor_ensemble(self, state: torch.Tensor, num_members: int) -> torch.Tensor:

        expanded_states = state.unsqueeze(1).expand(-1, num_members, *state.shape[1:])

        if self.method == "gaussian":
            noise = torch.randn_like(expanded_states) * self.noise_std
            return expanded_states + noise
        elif self.method == "dropout":
            # Apply different dropout patterns to each member
            result = []
            for i in range(num_members):
                # Use different random seed for each member
                dropout_mask = torch.rand_like(state) > 0.1  # 10% dropout rate
                perturbed = state * dropout_mask
                result.append(perturbed)
            return torch.stack(result, dim=1)
        else:  # perturbation
            # Apply small random perturbations
            perturbations = torch.randn_like(expanded_states) * self.noise_std
            return expanded_states + perturbations

    def _generate_graph_ensemble(
        self, state: Union[Data, HeteroData], num_members: int
    ) -> Union[Data, HeteroData]:

        # For graph-based states, we replicate the structure and add noise to node/edge features
        if isinstance(state, HeteroData):
            # Handle heterogeneous graphs
            result = HeteroData()

            # Copy graph structure for each ensemble member
            for node_type in state.node_types:
                if hasattr(state[node_type], "x"):
                    node_features = state[node_type].x
                    # Get batch size for reference
                    _ = node_features.size(0) if node_features.dim() > 1 else 1

                    if node_features.dim() == 2:
                        # [num_nodes, features] -> [num_nodes, num_members, features]
                        expanded_features = node_features.unsqueeze(1).expand(-1, num_members, -1)

                        if self.method == "gaussian":
                            noise = torch.randn_like(expanded_features) * self.noise_std
                            result[node_type].x = expanded_features + noise
                        else:
                            perturbations = torch.randn_like(expanded_features) * self.noise_std
                            result[node_type].x = expanded_features + perturbations
                    else:
                        result[node_type].x = node_features

            # Copy edge attributes similarly
            for edge_type in state.edge_types:
                if (
                    hasattr(state[edge_type], "edge_attr")
                    and state[edge_type].edge_attr is not None
                ):
                    edge_attrs = state[edge_type].edge_attr
                    if edge_attrs.dim() == 2:
                        expanded_attrs = edge_attrs.unsqueeze(1).expand(-1, num_members, -1)

                        if self.method == "gaussian":
                            noise = torch.randn_like(expanded_attrs) * self.noise_std
                            result[edge_type].edge_attr = expanded_attrs + noise
                        else:
                            perturbations = torch.randn_like(expanded_attrs) * self.noise_std
                            result[edge_type].edge_attr = expanded_attrs + perturbations
                    else:
                        result[edge_type].edge_attr = edge_attrs

                # Copy edge indices
                if hasattr(state[edge_type], "edge_index"):
                    result[edge_type].edge_index = state[edge_type].edge_index

            return result
        else:
            # Handle homogeneous graphs
            result = Data()

            # Copy node features with ensemble dimension
            if hasattr(state, "x") and state.x is not None:
                node_features = state.x
                if node_features.dim() == 2:
                    # [num_nodes, features] -> [num_nodes, num_members, features]
                    expanded_features = node_features.unsqueeze(1).expand(-1, num_members, -1)

                    if self.method == "gaussian":
                        noise = torch.randn_like(expanded_features) * self.noise_std
                        result.x = expanded_features + noise
                    else:
                        perturbations = torch.randn_like(expanded_features) * self.noise_std
                        result.x = expanded_features + perturbations
                else:
                    result.x = state.x

            # Copy edge attributes
            if hasattr(state, "edge_attr") and state.edge_attr is not None:
                edge_attrs = state.edge_attr
                if edge_attrs.dim() == 2:
                    expanded_attrs = edge_attrs.unsqueeze(1).expand(-1, num_members, -1)

                    if self.method == "gaussian":
                        noise = torch.randn_like(expanded_attrs) * self.noise_std
                        result.edge_attr = expanded_attrs + noise
                    else:
                        perturbations = torch.randn_like(expanded_attrs) * self.noise_std
                        result.edge_attr = expanded_attrs + perturbations
                else:
                    result.edge_attr = state.edge_attr

            # Copy edge indices
            if hasattr(state, "edge_index"):
                result.edge_index = state.edge_index

            return result
