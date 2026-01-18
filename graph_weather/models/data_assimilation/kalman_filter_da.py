from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
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


class KalmanFilterDA(DataAssimilationBase):

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)

        self.ensemble_size = self.config.get("ensemble_size", 20)
        self.inflation_factor = self.config.get("inflation_factor", 1.1)
        self.observation_error_std = self.config.get("observation_error_std", 0.1)
        self.background_error_std = self.config.get("background_error_std", 0.5)

        # Ensemble generator for creating diverse ensemble members
        self.ensemble_generator = EnsembleGenerator(
            noise_std=self.background_error_std, method="gaussian"
        )

        # Learnable parameters for adaptive inflation
        self.adaptive_inflation = self.config.get("adaptive_inflation", True)
        if self.adaptive_inflation:
            self.inflation_param = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if ensemble_members is None:
            ensemble = self.initialize_ensemble(state_in, self.ensemble_size)
        else:
            ensemble = ensemble_members

        # Perform assimilation
        updated_ensemble = self.assimilate(ensemble, observations)

        # Return analysis state (mean of ensemble)
        return self._compute_analysis(updated_ensemble)

    def initialize_ensemble(
        self, background_state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:

        return self.ensemble_generator(background_state, num_members)

    def assimilate(
        self, ensemble: Union[torch.Tensor, Data, HeteroData], observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if isinstance(ensemble, torch.Tensor):
            return self._assimilate_tensor_ensemble(ensemble, observations)
        elif isinstance(ensemble, (Data, HeteroData)):
            return self._assimilate_graph_ensemble(ensemble, observations)
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")

    def _assimilate_tensor_ensemble(
        self, ensemble: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:

        batch_size, num_members = ensemble.size(0), ensemble.size(1)

        # Reshape ensemble for computation: [batch_size * num_members, ...]
        orig_shape = ensemble.shape[2:]  # Original feature dimensions
        # Reshape ensemble for computation: [batch_size * num_members, ...]
        _ = ensemble.view(batch_size * num_members, *orig_shape)  # Not used but kept for clarity

        # Compute ensemble mean and perturbations
        ensemble_mean = torch.mean(ensemble, dim=1, keepdim=True)  # [batch_size, 1, ...]
        ensemble_perts = ensemble - ensemble_mean  # [batch_size, num_members, ...]

        # Apply multiplicative inflation
        if self.adaptive_inflation:
            inflation = 1.0 + torch.tanh(self.inflation_param)
        else:
            inflation = self.inflation_factor

        ensemble_perts = ensemble_perts * inflation
        inflated_ensemble = ensemble_mean + ensemble_perts  # [batch_size, num_members, ...]

        # Compute observation operator H (identity for direct observations)
        # For simplicity, assume observations are extracted from state features
        # In practice, H would be a more complex operator
        # Here we assume obs_dim is a subset of state features

        # Compute ensemble statistics for Kalman gain calculation
        # First, create a simplified observation operator for demonstration
        # In real applications, H would map state space to observation space
        state_dim = int(torch.prod(torch.tensor(orig_shape)))
        obs_dim = observations.size(1)

        # Simplified approach: extract first 'obs_dim' features as observations
        if state_dim >= obs_dim:
            # Take first obs_dim features as pseudo-observations
            ensemble_obs = inflated_ensemble[:, :, :obs_dim]  # [batch_size, num_members, obs_dim]
        else:
            # If state is smaller than obs space, expand using repetition
            reps = obs_dim // state_dim + (1 if obs_dim % state_dim > 0 else 0)
            expanded = inflated_ensemble[:, :, :state_dim].repeat(1, 1, reps)
            ensemble_obs = expanded[:, :, :obs_dim]  # [batch_size, num_members, obs_dim]

        # Compute ensemble mean in observation space
        obs_mean = torch.mean(ensemble_obs, dim=1, keepdim=True)  # [batch_size, 1, obs_dim]
        obs_perts = ensemble_obs - obs_mean  # [batch_size, num_members, obs_dim]

        # Compute error covariance matrices
        # Background error covariance in observation space: P_b = (1/(N-1)) * HH^T
        # where H is ensemble perturbations in observation space
        # Observation error covariance: R
        R = (self.observation_error_std**2) * torch.eye(
            obs_dim, device=observations.device, dtype=observations.dtype
        ).unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, obs_dim, obs_dim]

        # Ensemble covariance in observation space
        # P_hh = (1/(N-1)) * H @ H^T
        # where H is obs perturbations
        P_hh = torch.matmul(
            obs_perts.transpose(-2, -1),  # [batch_size, obs_dim, num_members]
            obs_perts,  # [batch_size, num_members, obs_dim]
        ) / (
            num_members - 1
        )  # [batch_size, obs_dim, obs_dim]

        # Innovation covariance: S = P_hh + R
        S = P_hh + R  # [batch_size, obs_dim, obs_dim]

        # Kalman gain: K = P_xh @ S^(-1)
        # P_xh is cross-covariance between state and obs space
        # For simplicity, assuming P_xh = P_bh
        # (same as state-obs cross-cov)
        state_perts = inflated_ensemble - torch.mean(inflated_ensemble, dim=1, keepdim=True)

        P_xh = torch.matmul(
            state_perts.transpose(-2, -1),  # [batch_size, state_dim, num_members]
            obs_perts,  # [batch_size, num_members, obs_dim]
        ) / (
            num_members - 1
        )  # [batch_size, state_dim, obs_dim]

        # Compute Kalman gain
        # K = P_xh @ inv(S)
        S_inv = torch.inverse(S)  # [batch_size, obs_dim, obs_dim]
        K = torch.matmul(P_xh, S_inv)  # [batch_size, state_dim, obs_dim]

        # Innovation: y - H*x_b
        innovation = observations.unsqueeze(1) - obs_mean  # [batch_size, 1, obs_dim]

        # Update ensemble mean: x_a = x_b + K*(y - H*x_b)
        dx = torch.matmul(K, innovation.transpose(-2, -1)).squeeze(-1)  # [batch_size, state_dim]

        # Add correction to each ensemble member
        ensemble_mean_orig = inflated_ensemble.mean(dim=1)  # [batch_size, state_dim]
        updated_mean = ensemble_mean_orig + dx  # [batch_size, state_dim]
        
        # Expand dx to match ensemble dimensions and apply to each member
        dx_expanded = dx.unsqueeze(1).expand(-1, num_members, -1)  # [batch_size, num_members, state_dim]
        updated_ensemble = inflated_ensemble + dx_expanded

        return updated_ensemble

    def _assimilate_graph_ensemble(
        self, ensemble: Union[Data, HeteroData], observations: torch.Tensor
    ) -> Union[Data, HeteroData]:

        # For graph-based ensemble, we need to handle node and edge features appropriately
        # This is a simplified implementation focusing on node features
        if isinstance(ensemble, HeteroData):
            # Handle heterogeneous graph ensemble
            result = HeteroData()

            for node_type in ensemble.node_types:
                if hasattr(ensemble[node_type], "x") and ensemble[node_type].x is not None:
                    node_features = ensemble[node_type].x

                    # Assuming node_features has shape [num_nodes, num_members, features]
                    if node_features.dim() == 3:
                        num_nodes, num_members, feat_dim = node_features.shape

                        # Reshape for ensemble operations: [num_nodes, features, num_members]
                        # Reshape for ensemble operations
                        node_features = node_features.transpose(1, 2)
                        # [num_nodes, features, num_members]

                        # Perform ensemble Kalman filter operations
                        # Compute ensemble mean
                        # Compute ensemble mean
                        ens_mean = torch.mean(node_features, dim=2, keepdim=True)
                        # [num_nodes, features, 1]

                        # Compute perturbations
                        ens_perts = node_features - ens_mean  # [num_nodes, features, num_members]

                        # Apply inflation
                        if self.adaptive_inflation:
                            inflation = 1.0 + torch.tanh(self.inflation_param)
                        else:
                            inflation = self.inflation_factor

                        ens_perts = ens_perts * inflation
                        inflated_ens = ens_mean + ens_perts  # [num_nodes, features, num_members]

                        # Transpose back to original format: [num_nodes, num_members, features]
                        result[node_type].x = inflated_ens.transpose(1, 2)
                    else:
                        result[node_type].x = ensemble[node_type].x
                else:
                    # Copy node attributes that don't have features
                    for key, value in ensemble[node_type].items():
                        if key != "x":
                            setattr(result[node_type], key, value)

            # Copy edge attributes similarly
            for edge_type in ensemble.edge_types:
                for key, value in ensemble[edge_type].items():
                    setattr(result[edge_type], key, value)

            return result
        else:
            # Handle homogeneous graph ensemble
            result = Data()

            if hasattr(ensemble, "x") and ensemble.x is not None:
                node_features = ensemble.x

                # Assuming node_features has shape [num_nodes, num_members, features]
                if node_features.dim() == 3:
                    num_nodes, num_members, feat_dim = node_features.shape

                    # Reshape for ensemble operations
                    node_features = node_features.transpose(1, 2)
                    # [num_nodes, features, num_members]

                    # Perform ensemble Kalman filter operations
                    # Compute ensemble mean
                    ens_mean = torch.mean(node_features, dim=2, keepdim=True)
                    # [num_nodes, features, 1]

                    # Compute perturbations
                    ens_perts = node_features - ens_mean  # [num_nodes, features, num_members]

                    # Apply inflation
                    if self.adaptive_inflation:
                        inflation = 1.0 + torch.tanh(self.inflation_param)
                    else:
                        inflation = self.inflation_factor

                    ens_perts = ens_perts * inflation
                    inflated_ens = ens_mean + ens_perts  # [num_nodes, features, num_members]

                    # Transpose back to original format: [num_nodes, num_members, features]
                    result.x = inflated_ens.transpose(1, 2)
                else:
                    result.x = ensemble.x

            # Copy other attributes
            for key, value in ensemble.items():
                if key != "x":
                    setattr(result, key, value)

            return result

    def _compute_analysis(
        self, ensemble: Union[torch.Tensor, Data, HeteroData]
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if isinstance(ensemble, torch.Tensor):
            # Return mean across ensemble dimension (dim=1)
            return torch.mean(ensemble, dim=1)
        elif isinstance(ensemble, HeteroData):
            result = HeteroData()

            for node_type in ensemble.node_types:
                if hasattr(ensemble[node_type], "x") and ensemble[node_type].x is not None:
                    node_features = ensemble[node_type].x
                    if node_features.dim() == 3:  # [num_nodes, num_members, features]
                        # Mean across ensemble dimension (dim=1)
                        result[node_type].x = torch.mean(node_features, dim=1)
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
                    if (
                        key != "edge_attr"
                        or ensemble[edge_type].edge_attr is None
                        or ensemble[edge_type].edge_attr.dim() != 3
                    ):
                        setattr(result[edge_type], key, value)
                    else:
                        # Average edge attributes across ensemble
                        edge_attr = ensemble[edge_type].edge_attr
                        if edge_attr.dim() == 3:  # [num_edges, num_members, features]
                            result[edge_type].edge_attr = torch.mean(edge_attr, dim=1)
                        else:
                            result[edge_type].edge_attr = edge_attr

            return result
        elif isinstance(ensemble, Data):
            result = Data()

            if hasattr(ensemble, "x") and ensemble.x is not None:
                node_features = ensemble.x
                if node_features.dim() == 3:  # [num_nodes, num_members, features]
                    # Mean across ensemble dimension (dim=1)
                    result.x = torch.mean(node_features, dim=1)
                else:
                    result.x = ensemble.x

            # Copy other attributes
            for key, value in ensemble.items():
                if key != "x":
                    setattr(result, key, value)

            return result
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")