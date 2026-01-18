from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import with fallbacks to handle different execution contexts
try:
    # For relative import when used as part of package
    from .data_assimilation_base import DataAssimilationBase, EnsembleGenerator
except ImportError:
    try:
        # For absolute import when used as standalone
        from graph_weather.models.data_assimilation.data_assimilation_base import DataAssimilationBase, EnsembleGenerator
    except ImportError:
        # For direct execution in isolated context
        import sys
        import os
        import importlib.util
        
        # Load the base module dynamically
        base_path = os.path.join(os.path.dirname(__file__), 'data_assimilation_base.py')
        spec = importlib.util.spec_from_file_location('data_assimilation_base', base_path)
        base_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_module)
        
        DataAssimilationBase = base_module.DataAssimilationBase
        EnsembleGenerator = base_module.EnsembleGenerator
from torch_geometric.data import Data, HeteroData


class VariationalDA(DataAssimilationBase):

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)

        self.iterations = self.config.get("iterations", 10)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.regularization_weight = self.config.get("regularization_weight", 0.1)
        self.background_error_std = self.config.get("background_error_std", 0.5)
        self.observation_error_std = self.config.get("observation_error_std", 0.1)

        # Ensemble generator for initial ensemble
        self.ensemble_generator = EnsembleGenerator(
            noise_std=self.background_error_std * 0.1, method="gaussian"
        )

        # Learnable parameters for adaptive weighting
        self.bg_weight = nn.Parameter(torch.tensor(1.0))
        self.obs_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Data, HeteroData]:

        # Initialize analysis state from background state
        if isinstance(state_in, torch.Tensor):
            analysis_state = state_in.clone().detach().requires_grad_(True)
        elif isinstance(state_in, (Data, HeteroData)):
            analysis_state = self._clone_graph_state(state_in).requires_grad_(True)
        else:
            raise TypeError(f"Unsupported state type: {type(state_in)}")

        # Optimize using gradient descent
        for i in range(self.iterations):
            # Zero gradients
            if analysis_state.grad is not None:
                analysis_state.grad.zero_()

            # Compute cost function
            cost = self._compute_cost_function(analysis_state, state_in, observations)

            # Backpropagate
            cost.backward()

            # Update analysis state (gradient descent step)
            with torch.no_grad():
                if isinstance(analysis_state, torch.Tensor):
                    analysis_state -= self.learning_rate * analysis_state.grad
                else:
                    # For graph states, update node features
                    if hasattr(analysis_state, "x") and analysis_state.x is not None:
                        analysis_state.x -= self.learning_rate * analysis_state.x.grad
                        analysis_state.x = analysis_state.x.detach().requires_grad_(True)

        # Return the optimized analysis state
        return analysis_state.detach()

    def initialize_ensemble(
        self, background_state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:

        return self.ensemble_generator(background_state, num_members)

    def assimilate(
        self, ensemble: Union[torch.Tensor, Data, HeteroData], observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:

        # For variational DA, we typically optimize a single analysis state
        # rather than updating each ensemble member individually
        # Here we'll apply the optimization to each member in the ensemble

        if isinstance(ensemble, torch.Tensor):
            return self._assimilate_tensor_ensemble(ensemble, observations)
        elif isinstance(ensemble, (Data, HeteroData)):
            return self._assimilate_graph_ensemble(ensemble, observations)
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")

    def _assimilate_tensor_ensemble(
        self, ensemble: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:

        # Extract ensemble dimensions
        _ = ensemble.size(0)  # batch_size
        num_members = ensemble.size(1)  # num_members

        # Process each ensemble member separately
        updated_ensemble = []

        for member_idx in range(num_members):
            # Select a member to optimize
            member = ensemble[:, member_idx].clone().detach().requires_grad_(True)
            background_member = ensemble[:, member_idx].detach()

            # Perform optimization for this member
            for i in range(self.iterations):
                if member.grad is not None:
                    member.grad.zero_()

                # Compute cost function for this member
                cost = self._compute_cost_function(member, background_member, observations)

                # Backpropagate
                cost.backward()

                # Update member
                with torch.no_grad():
                    member -= self.learning_rate * member.grad

            updated_ensemble.append(member.detach())

        return torch.stack(updated_ensemble, dim=1)

    def _assimilate_graph_ensemble(
        self, ensemble: Union[Data, HeteroData], observations: torch.Tensor
    ) -> Union[Data, HeteroData]:

        # For graph-based ensemble, we need to handle the optimization differently
        # This is a simplified implementation that optimizes the node features

        if isinstance(ensemble, HeteroData):
            # Handle heterogeneous graph ensemble
            result = HeteroData()

            for node_type in ensemble.node_types:
                if hasattr(ensemble[node_type], "x") and ensemble[node_type].x is not None:
                    node_features = ensemble[node_type].x

                    # If node_features has ensemble dimension [num_nodes, num_members, features]
                    if node_features.dim() == 3:
                        # Extract node dimensions
                        _ = node_features.shape  # num_nodes, num_members, feat_dim
                        num_members = node_features.size(1)

                        updated_members = []
                        for member_idx in range(num_members):
                            # Process each ensemble member
                            member = (
                                node_features[:, member_idx].clone().detach().requires_grad_(True)
                            )
                            background_member = node_features[:, member_idx].detach()

                            # Perform optimization for this member
                            for i in range(self.iterations):
                                if member.grad is not None:
                                    member.grad.zero_()

                                # Compute cost function for this member
                                # Simplified: treat as tensor optimization
                                cost = self._compute_tensor_cost_function(
                                    member, background_member, observations
                                )

                                # Backpropagate
                                cost.backward()

                                # Update member
                                with torch.no_grad():
                                    member -= self.learning_rate * member.grad

                            updated_members.append(member.detach())

                        result[node_type].x = torch.stack(updated_members, dim=1)
                    else:
                        result[node_type].x = ensemble[node_type].x
                else:
                    # Copy other node attributes
                    for key, value in ensemble[node_type].items():
                        if key != "x":
                            setattr(result[node_type], key, value)

            # Copy edge attributes
            for edge_type in ensemble.edge_types:
                for key, value in ensemble[edge_type].items():
                    setattr(result[edge_type], key, value)

            return result
        else:
            # Handle homogeneous graph ensemble
            result = Data()

            if hasattr(ensemble, "x") and ensemble.x is not None:
                node_features = ensemble.x

                # If node_features has ensemble dimension [num_nodes, num_members, features]
                if node_features.dim() == 3:
                    # Extract node dimensions
                    _ = node_features.shape  # num_nodes, num_members, feat_dim
                    num_members = node_features.size(1)

                    updated_members = []
                    for member_idx in range(num_members):
                        # Process each ensemble member
                        member = node_features[:, member_idx].clone().detach().requires_grad_(True)
                        background_member = node_features[:, member_idx].detach()

                        # Perform optimization for this member
                        for i in range(self.iterations):
                            if member.grad is not None:
                                member.grad.zero_()

                            # Compute cost function for this member
                            cost = self._compute_tensor_cost_function(
                                member, background_member, observations
                            )

                            # Backpropagate
                            cost.backward()

                            # Update member
                            with torch.no_grad():
                                member -= self.learning_rate * member.grad

                        updated_members.append(member.detach())

                    result.x = torch.stack(updated_members, dim=1)
                else:
                    result.x = ensemble.x

            # Copy other attributes
            for key, value in ensemble.items():
                if key != "x":
                    setattr(result, key, value)

            return result

    def _compute_cost_function(
        self,
        analysis_state: Union[torch.Tensor, Data, HeteroData],
        background_state: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
    ) -> torch.Tensor:

        if isinstance(analysis_state, torch.Tensor):
            return self._compute_tensor_cost_function(
                analysis_state, background_state, observations
            )
        elif isinstance(analysis_state, (Data, HeteroData)):
            return self._compute_graph_cost_function(analysis_state, background_state, observations)
        else:
            raise TypeError(f"Unsupported state type: {type(analysis_state)}")

    def _compute_tensor_cost_function(
        self,
        analysis_state: torch.Tensor,
        background_state: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:

        # Ensure same shape for comparison
        if analysis_state.shape != background_state.shape:
            # Try to reshape or broadcast appropriately
            if len(analysis_state.shape) > len(background_state.shape):
                # analysis_state might have extra ensemble dimension
                if analysis_state.shape[1:] == background_state.shape:
                    analysis_state = analysis_state.mean(dim=1)  # Average over ensemble dim
            elif len(background_state.shape) > len(analysis_state.shape):
                if background_state.shape[1:] == analysis_state.shape:
                    background_state = background_state.mean(dim=1)  # Average over ensemble dim

        # Background term: (x - x_b)^T B^{-1} (x - x_b)
        bg_diff = analysis_state - background_state
        bg_error_var = self.background_error_std**2
        bg_term = torch.sum(bg_diff**2) / bg_error_var

        # Observation term: (y - Hx)^T R^{-1} (y - Hx)
        # Assume H is identity for direct observation of state variables
        # Or extract observation-space features from state
        state_flat = analysis_state.view(analysis_state.size(0), -1)  # Flatten spatial dims
        obs_dim = observations.size(1)
        state_dim = state_flat.size(1)

        if state_dim >= obs_dim:
            # Extract first obs_dim features as observed quantities
            state_obs = state_flat[:, :obs_dim]
        else:
            # If state is smaller than obs space, expand using repetition
            reps = obs_dim // state_dim + (1 if obs_dim % state_dim > 0 else 0)
            expanded = state_flat.repeat(1, reps)
            state_obs = expanded[:, :obs_dim]

        obs_diff = observations - state_obs
        obs_error_var = self.observation_error_std**2
        obs_term = torch.sum(obs_diff**2) / obs_error_var

        # Regularization term to prevent overfitting
        reg_term = self.regularization_weight * torch.sum(analysis_state**2)

        # Combine terms with learnable weights
        bg_w = torch.clamp(F.softplus(self.bg_weight), min=0.1, max=10.0)
        obs_w = torch.clamp(F.softplus(self.obs_weight), min=0.1, max=10.0)

        total_cost = bg_w * bg_term + obs_w * obs_term + reg_term

        return total_cost

    def _compute_graph_cost_function(
        self,
        analysis_state: Union[Data, HeteroData],
        background_state: Union[Data, HeteroData],
        observations: torch.Tensor,
    ) -> torch.Tensor:

        # Extract node features for cost computation
        if hasattr(analysis_state, "x") and analysis_state.x is not None:
            analysis_features = analysis_state.x
        else:
            # For HeteroData, use first node type
            if isinstance(analysis_state, HeteroData):
                first_node_type = next(iter(analysis_state.node_types))
                analysis_features = analysis_state[first_node_type].x
            else:
                raise ValueError("Graph state has no node features")

        if hasattr(background_state, "x") and background_state.x is not None:
            background_features = background_state.x
        else:
            if isinstance(background_state, HeteroData):
                first_node_type = next(iter(background_state.node_types))
                background_features = background_state[first_node_type].x
            else:
                raise ValueError("Background state has no node features")

        # Make sure both have same shape for comparison
        if analysis_features.shape != background_features.shape:
            if analysis_features.dim() > background_features.dim():
                # analysis might have ensemble dimension
                if analysis_features.shape[1:] == background_features.shape:
                    analysis_features = analysis_features.mean(dim=1)
            elif background_features.dim() > analysis_features.dim():
                if background_features.shape[1:] == analysis_features.shape:
                    background_features = background_features.mean(dim=1)

        # Background term
        bg_diff = analysis_features - background_features
        bg_error_var = self.background_error_std**2
        bg_term = torch.sum(bg_diff**2) / bg_error_var

        # Observation term
        # Map node features to observation space
        features_flat = analysis_features.view(analysis_features.size(0), -1)
        obs_dim = observations.size(1)
        features_dim = features_flat.size(1)

        if features_dim >= obs_dim:
            features_obs = features_flat[:, :obs_dim]
        else:
            reps = obs_dim // features_dim + (1 if obs_dim % features_dim > 0 else 0)
            expanded = features_flat.repeat(1, reps)
            features_obs = expanded[:, :obs_dim]

        obs_diff = observations - features_obs
        obs_error_var = self.observation_error_std**2
        obs_term = torch.sum(obs_diff**2) / obs_error_var

        # Regularization term
        reg_term = self.regularization_weight * torch.sum(analysis_features**2)

        # Combine terms with learnable weights
        bg_w = torch.clamp(F.softplus(self.bg_weight), min=0.1, max=10.0)
        obs_w = torch.clamp(F.softplus(self.obs_weight), min=0.1, max=10.0)

        total_cost = bg_w * bg_term + obs_w * obs_term + reg_term

        return total_cost

    def _clone_graph_state(self, graph_state: Union[Data, HeteroData]) -> Union[Data, HeteroData]:

        if isinstance(graph_state, HeteroData):
            cloned = HeteroData()
            for node_type in graph_state.node_types:
                for key, value in graph_state[node_type].items():
                    setattr(
                        cloned[node_type], key, value.clone() if torch.is_tensor(value) else value
                    )
            for edge_type in graph_state.edge_types:
                for key, value in graph_state[edge_type].items():
                    setattr(
                        cloned[edge_type], key, value.clone() if torch.is_tensor(value) else value
                    )
            return cloned
        else:
            cloned = Data()
            for key, value in graph_state.items():
                setattr(cloned, key, value.clone() if torch.is_tensor(value) else value)
            return cloned

    def _compute_analysis(
        self, ensemble: Union[torch.Tensor, Data, HeteroData]
    ) -> Union[torch.Tensor, Data, HeteroData]:

        # In variational DA, the analysis is the optimized state itself
        # This method is kept for interface consistency
        if isinstance(ensemble, torch.Tensor):
            return torch.mean(ensemble, dim=1) if ensemble.dim() > 2 else ensemble
        elif isinstance(ensemble, HeteroData):
            result = HeteroData()
            for node_type in ensemble.node_types:
                if hasattr(ensemble[node_type], "x") and ensemble[node_type].x is not None:
                    node_features = ensemble[node_type].x
                    if node_features.dim() == 3:  # [num_nodes, num_members, features]
                        result[node_type].x = torch.mean(node_features, dim=1)
                    else:
                        result[node_type].x = node_features
                else:
                    for key, value in ensemble[node_type].items():
                        if key != "x":
                            setattr(result[node_type], key, value)
            for edge_type in ensemble.edge_types:
                for key, value in ensemble[edge_type].items():
                    setattr(result[edge_type], key, value)
            return result
        elif isinstance(ensemble, Data):
            result = Data()
            if hasattr(ensemble, "x") and ensemble.x is not None:
                node_features = ensemble.x
                if node_features.dim() == 3:  # [num_nodes, num_members, features]
                    result.x = torch.mean(node_features, dim=1)
                else:
                    result.x = ensemble.x
            for key, value in ensemble.items():
                if key != "x":
                    setattr(result, key, value)
            return result
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")