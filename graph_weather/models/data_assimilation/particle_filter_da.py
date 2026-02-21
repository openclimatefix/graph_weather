from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .data_assimilation_base import DataAssimilationBase, EnsembleGenerator
from torch_geometric.data import Data, HeteroData


class ParticleFilterDA(DataAssimilationBase):

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)

        self.num_particles = self.config.get("num_particles", 100)
        # fraction of total particles
        self.resample_threshold = self.config.get("resample_threshold", 0.5)
        self.observation_error_std = self.config.get("observation_error_std", 0.1)
        self.process_noise_std = self.config.get("process_noise_std", 0.05)

        # Ensemble generator for creating diverse particles
        self.particle_generator = EnsembleGenerator(
            noise_std=self.process_noise_std, method="gaussian"
        )

        # Temperature parameter for likelihood computation
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if ensemble_members is None:
            particles = self.initialize_ensemble(state_in, self.num_particles)
        else:
            particles = ensemble_members

        # Perform assimilation
        updated_particles = self.assimilate(particles, observations)

        # Return analysis state (weighted average of particles)
        return self._compute_analysis(updated_particles)

    def initialize_ensemble(
        self, background_state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:

        return self.particle_generator(background_state, num_members)

    def assimilate(
        self, ensemble: Union[torch.Tensor, Data, HeteroData], observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if isinstance(ensemble, torch.Tensor):
            return self._assimilate_tensor_particles(ensemble, observations)
        elif isinstance(ensemble, (Data, HeteroData)):
            return self._assimilate_graph_particles(ensemble, observations)
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")

    def _assimilate_tensor_particles(
        self, particles: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:

        # Extract batch and particle dimensions
        _ = particles.size(0)  # batch_size
        _ = particles.size(1)  # num_particles

        # Compute log-likelihood weights for each particle
        log_weights = self._compute_log_likelihood(particles, observations)

        # Normalize weights using log-sum-exp trick for numerical stability
        max_log_weight = torch.max(log_weights, dim=1, keepdim=True)[0]  # [batch_size, 1, ...]
        weights = torch.exp(log_weights - max_log_weight)

        # Normalize weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-12)

        # Check effective sample size and resample if needed
        _ = 1.0 / torch.sum(weights**2, dim=1)  # [batch_size, ...]  # effective_size
        _ = self.resample_threshold * particles.size(1)  # threshold

        # Resample particles based on weights
        resampled_particles = self._resample_particles(particles, weights)

        # Add small amount of noise to prevent particle degeneracy
        noise_scale = self.process_noise_std * 0.1
        noise = torch.randn_like(resampled_particles) * noise_scale
        resampled_particles = resampled_particles + noise

        return resampled_particles

    def _compute_log_likelihood(
        self, particles: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:

        # Extract particle dimensions
        _ = particles.size(0)  # batch_size
        _ = particles.size(1)  # num_particles

        # Map particles to observation space
        particle_obs = self._map_particles_to_observation_space(particles, observations.size(1))

        # Compute log-likelihood using common helper
        return self._compute_log_likelihood_from_obs(particle_obs, observations)

    def _map_particles_to_observation_space(self, particles: torch.Tensor, obs_dim: int) -> torch.Tensor:
        """Map particles to observation space by extracting or expanding features."""
        state_dim = int(torch.prod(torch.tensor(particles.shape[2:])))

        if state_dim >= obs_dim:
            # Take first obs_dim features as pseudo-observations
            particle_obs = particles[:, :, :obs_dim]  # [batch_size, num_particles, obs_dim]
        else:
            # If state is smaller than obs space, expand using repetition
            reps = obs_dim // state_dim + (1 if obs_dim % state_dim > 0 else 0)
            expanded = particles[:, :, :state_dim].repeat(1, 1, reps)
            particle_obs = expanded[:, :, :obs_dim]  # [batch_size, num_particles, obs_dim]
        
        return particle_obs

    def _compute_log_likelihood_from_obs(
        self, particle_obs: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-likelihood from particle observations and actual observations."""
        # Compute likelihood: p(y|x) ~ exp(-||y - Hx||^2 / (2*sigma^2))
        # Using temperature parameter for adaptability
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)

        # Observation error covariance
        obs_error_var = self.observation_error_std**2

        # Compute squared differences
        diff = particle_obs - observations.unsqueeze(1)  # [batch_size, num_particles, obs_dim]
        squared_diff = torch.sum(diff**2, dim=2)  # [batch_size, num_particles]

        # Compute log-likelihood
        log_likelihood = -squared_diff / (2 * obs_error_var * temp)

        # Expand back to match particle dimensions
        # The log_likelihood has shape [batch_size, num_particles],
        # so we need to broadcast it to match particle dimensions
        return log_likelihood.unsqueeze(-1).unsqueeze(-1)  # Add dims to match original shape

    def _resample_particles(self, particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

        # Extract batch and particle dimensions
        _ = particles.size(0)  # batch_size
        _ = particles.size(1)  # num_particles

        # Flatten weights for resampling (consider only the first weight dimension)
        flat_weights = weights.squeeze(-1).squeeze(-1)  # Remove extra dims if they exist

        # Perform systematic resampling using common helper
        indices = self._systematic_resampling_indices(flat_weights, particles.size(1))
        
        # Resample particles using common helper
        return self._resample_particles_by_indices(particles, indices)

    def _systematic_resampling_indices(self, weights: torch.Tensor, num_particles: int) -> torch.Tensor:
        """Generate resampling indices using systematic resampling."""
        device = weights.device
        dtype = weights.dtype
        batch_size = weights.size(0)

        # Create cumulative sum of weights
        cumsum_weights = torch.cumsum(weights, dim=1)  # [batch_size, num_particles]

        # Generate uniform samples for systematic resampling
        u = (
            torch.arange(num_particles, dtype=dtype, device=device)
            + torch.rand(batch_size, 1, device=device)
        ) / num_particles
        # [batch_size, num_particles]

        # Find indices of particles to select
        indices = torch.searchsorted(cumsum_weights, u.clamp(0, 1))  # [batch_size, num_particles]
        indices = torch.clamp(indices, 0, num_particles - 1)  # Ensure valid indices
        
        return indices

    def _resample_particles_by_indices(self, particles: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Resample particles using provided indices."""
        # Create output tensor
        resampled = torch.zeros_like(particles)

        # Resample particles for each batch
        for b in range(particles.size(0)):
            for i in range(particles.size(1)):
                resampled[b, i] = particles[b, indices[b, i]]

        return resampled

    def _assimilate_graph_particles(
        self, particles: Union[Data, HeteroData], observations: torch.Tensor
    ) -> Union[Data, HeteroData]:

        # For graph-based particles, we focus on resampling nodes based on weights
        if isinstance(particles, HeteroData):
            # Handle heterogeneous graph particles
            result = HeteroData()

            for node_type in particles.node_types:
                if hasattr(particles[node_type], "x") and particles[node_type].x is not None:
                    node_features = particles[node_type].x

                    # Assuming node_features has shape [num_nodes, num_particles, features]
                    if node_features.dim() == 3:
                        # Extract node dimensions for reference
                        _ = node_features.shape  # num_nodes, num_particles, feat_dim

                        # We need to compute weights for each particle across all nodes
                        # For simplicity, we'll use a simplified approach where weights
                        # are computed based on overall node feature statistics

                        # Compute mean features across nodes for each particle
                        # [num_particles, features]
                        particle_means = torch.mean(node_features, dim=0)

                        # Use first few features as pseudo-observations for weight computation
                        obs_dim = min(particle_means.size(1), observations.size(1))
                        particle_obs = particle_means[:, :obs_dim]  # [num_particles, obs_dim]

                        # Compute log-likelihood weights using common helper
                        obs_expanded = observations[0:1].expand(particle_means.size(0), -1)
                        log_likelihood = self._compute_log_likelihood_from_obs(
                            particle_obs.unsqueeze(0), obs_expanded
                        ).squeeze(0).squeeze(-1)

                        # Normalize weights
                        max_log_weight = torch.max(log_likelihood)
                        weights = torch.exp(log_likelihood - max_log_weight)
                        weights = weights / (torch.sum(weights) + 1e-12)

                        # Systematic resampling using common helper
                        # Reshape weights for batch dimension compatibility
                        weights_batched = weights.unsqueeze(0)  # [1, num_particles]
                        indices = self._systematic_resampling_indices(weights_batched, node_features.size(1))
                        indices = indices.squeeze(0)  # Remove batch dimension

                        # Resample particles using common helper
                        resampled_features = node_features[:, indices, :]

                        # Add small noise to prevent degeneracy
                        # Add small noise to prevent degeneracy
                        noise = torch.randn_like(resampled_features) * (
                            self.process_noise_std * 0.1
                        )
                        result[node_type].x = resampled_features + noise
                    else:
                        result[node_type].x = particles[node_type].x
                else:
                    # Copy other node attributes
                    for key, value in particles[node_type].items():
                        if key != "x":
                            setattr(result[node_type], key, value)

            # Copy edge attributes
            for edge_type in particles.edge_types:
                for key, value in particles[edge_type].items():
                    setattr(result[edge_type], key, value)

            return result
        else:
            # Handle homogeneous graph particles
            result = Data()

            if hasattr(particles, "x") and particles.x is not None:
                node_features = particles.x

                # Assuming node_features has shape [num_nodes, num_particles, features]
                if node_features.dim() == 3:
                    num_nodes, num_particles, feat_dim = node_features.shape

                    # Compute mean features across nodes for each particle
                    particle_means = torch.mean(node_features, dim=0)  # [num_particles, features]

                    # Use first few features as pseudo-observations for weight computation
                    obs_dim = min(particle_means.size(1), observations.size(1))
                    particle_obs = particle_means[:, :obs_dim]  # [num_particles, obs_dim]

                    # Compute log-likelihood weights using common helper
                    obs_expanded = observations[0:1].expand(num_particles, -1)
                    log_likelihood = self._compute_log_likelihood_from_obs(
                        particle_obs.unsqueeze(0), obs_expanded
                    ).squeeze(0).squeeze(-1)

                    # Normalize weights
                    max_log_weight = torch.max(log_likelihood)
                    weights = torch.exp(log_likelihood - max_log_weight)
                    weights = weights / (torch.sum(weights) + 1e-12)

                    # Systematic resampling using common helper
                    # Reshape weights for batch dimension compatibility
                    weights_batched = weights.unsqueeze(0)  # [1, num_particles]
                    indices = self._systematic_resampling_indices(weights_batched, num_particles)
                    indices = indices.squeeze(0)  # Remove batch dimension

                    # Resample particles using common helper
                    resampled_features = node_features[:, indices, :]

                    # Add small noise to prevent degeneracy
                    # Add small noise to prevent degeneracy
                    noise = torch.randn_like(resampled_features) * (self.process_noise_std * 0.1)
                    result.x = resampled_features + noise
                else:
                    result.x = particles.x

            # Copy other attributes
            for key, value in particles.items():
                if key != "x":
                    setattr(result, key, value)

            return result

    def _compute_analysis(
        self, ensemble: Union[torch.Tensor, Data, HeteroData]
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if isinstance(ensemble, torch.Tensor):
            # Return mean across particle dimension (dim=1)
            return torch.mean(ensemble, dim=1)
        elif isinstance(ensemble, (Data, HeteroData)):
            return self._compute_analysis_graph(ensemble)
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")

    def _compute_analysis_graph(
        self, ensemble: Union[Data, HeteroData]
    ) -> Union[Data, HeteroData]:
        """Compute analysis for graph ensembles with shared logic for both Data and HeteroData."""
        if isinstance(ensemble, HeteroData):
            result = HeteroData()
            node_types = ensemble.node_types
            edge_types = ensemble.edge_types
        else:
            result = Data()
            node_types = [None]  # Single node type for homogeneous graphs
            edge_types = [None]  # Single edge type for homogeneous graphs

        # Process node features
        for node_type in node_types:
            # Get the actual node data (different access for HeteroData vs Data)
            if isinstance(ensemble, HeteroData):
                node_data = ensemble[node_type]
                result_node_data = result[node_type]
            else:
                node_data = ensemble
                result_node_data = result
                node_type = None  # For homogeneous graphs
                
            if hasattr(node_data, "x") and node_data.x is not None:
                node_features = node_data.x
                if node_features.dim() == 3:  # [num_nodes, num_particles, features]
                    # Mean across particle dimension (dim=1)
                    result_node_data.x = torch.mean(node_features, dim=1)
                else:
                    result_node_data.x = node_data.x
            else:
                # Copy other node attributes
                for key, value in node_data.items():
                    if key != "x":
                        setattr(result_node_data, key, value)

        # Copy edge attributes
        if isinstance(ensemble, HeteroData):
            for edge_type in edge_types:
                for key, value in ensemble[edge_type].items():
                    setattr(result[edge_type], key, value)
        else:
            # For homogeneous graphs, copy edge attributes directly
            for key, value in ensemble.items():
                if key not in ["x"] and not key.startswith("edge"):
                    setattr(result, key, value)

        return result
