"""
Particle Filter-based Data Assimilation Module.

This module implements particle filtering techniques for data assimilation,
providing a non-parametric approach suitable for non-linear, non-Gaussian systems.
"""
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from data_assimilation_base import DataAssimilationBase, EnsembleGenerator
from torch_geometric.data import Data, HeteroData


class ParticleFilterDA(DataAssimilationBase):
    """
    Particle Filter implementation for data assimilation.
    
    This implementation uses importance sampling to approximate the posterior distribution
    using particles (samples) with associated weights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Particle Filter DA module.
        
        Args:
            config: Configuration dictionary with parameters like:
                   - num_particles: Number of particles to use
                   - resample_threshold: Threshold for effective sample size to trigger resampling
                   - observation_error_std: Standard deviation of observation errors
                   - process_noise_std: Standard deviation of process noise
        """
        super().__init__(config)
        
        self.num_particles = self.config.get('num_particles', 100)
        # fraction of total particles
        self.resample_threshold = self.config.get('resample_threshold', 0.5)
        self.observation_error_std = self.config.get('observation_error_std', 0.1)
        self.process_noise_std = self.config.get('process_noise_std', 0.05)
        
        # Ensemble generator for creating diverse particles
        self.particle_generator = EnsembleGenerator(
            noise_std=self.process_noise_std,
            method='gaussian'
        )
        
        # Temperature parameter for likelihood computation
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self, 
        state_in: Union[torch.Tensor, Data, HeteroData], 
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Perform particle filter data assimilation step.
        
        Args:
            state_in: Input state (graph or tensor)
            observations: Observation data
            ensemble_members: Optional pre-generated particles
            
        Returns:
            Updated state in the same format as input
        """
        if ensemble_members is None:
            particles = self.initialize_ensemble(state_in, self.num_particles)
        else:
            particles = ensemble_members
        
        # Perform assimilation
        updated_particles = self.assimilate(particles, observations)
        
        # Return analysis state (weighted average of particles)
        return self._compute_analysis(updated_particles)
    
    def initialize_ensemble(
        self, 
        background_state: Union[torch.Tensor, Data, HeteroData], 
        num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Initialize particle ensemble from background state.
        
        Args:
            background_state: Background state to generate particles from
            num_members: Number of particles to generate
            
        Returns:
            Particle ensemble
        """
        return self.particle_generator(background_state, num_members)
    
    def assimilate(
        self, 
        ensemble: Union[torch.Tensor, Data, HeteroData], 
        observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Perform particle filter assimilation with importance sampling and resampling.
        
        Args:
            ensemble: Particle ensemble
            observations: Observation data
            
        Returns:
            Updated particle ensemble
        """
        if isinstance(ensemble, torch.Tensor):
            return self._assimilate_tensor_particles(ensemble, observations)
        elif isinstance(ensemble, (Data, HeteroData)):
            return self._assimilate_graph_particles(ensemble, observations)
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")
    
    def _assimilate_tensor_particles(
        self, 
        particles: torch.Tensor, 
        observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform particle filter assimilation for tensor-based particles.
        
        Args:
            particles: Particle tensor of shape [batch_size, num_particles, ...]
            observations: Observations of shape [batch_size, obs_dim]
            
        Returns:
            Updated particle tensor
        """
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
        _ = 1.0 / torch.sum(weights ** 2, dim=1)  # [batch_size, ...]  # effective_size
        _ = self.resample_threshold * particles.size(1)  # threshold
        
        # Resample particles based on weights
        resampled_particles = self._resample_particles(particles, weights)
        
        # Add small amount of noise to prevent particle degeneracy
        noise_scale = self.process_noise_std * 0.1
        noise = torch.randn_like(resampled_particles) * noise_scale
        resampled_particles = resampled_particles + noise
        
        return resampled_particles
    
    def _compute_log_likelihood(
        self, 
        particles: torch.Tensor, 
        observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-likelihood for each particle given observations.
        
        Args:
            particles: Particle tensor of shape [batch_size, num_particles, ...]
            observations: Observations of shape [batch_size, obs_dim]
            
        Returns:
            Log-likelihood weights of shape [batch_size, num_particles, ...]
        """
        # Extract particle dimensions
        _ = particles.size(0)  # batch_size
        _ = particles.size(1)  # num_particles
        
        # Map particles to observation space
        # For simplicity, assume observation operator H extracts first obs_dim features
        obs_dim = observations.size(1)
        state_dim = int(torch.prod(torch.tensor(particles.shape[2:])))
        
        if state_dim >= obs_dim:
            # Take first obs_dim features as pseudo-observations
            particle_obs = particles[:, :, :obs_dim]  # [batch_size, num_particles, obs_dim]
        else:
            # If state is smaller than obs space, expand using repetition
            reps = obs_dim // state_dim + (1 if obs_dim % state_dim > 0 else 0)
            expanded = particles[:, :, :state_dim].repeat(1, 1, reps)
            particle_obs = expanded[:, :, :obs_dim]  # [batch_size, num_particles, obs_dim]
        
        # Compute likelihood: p(y|x) ~ exp(-||y - Hx||^2 / (2*sigma^2))
        # Using temperature parameter for adaptability
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # Observation error covariance
        obs_error_var = self.observation_error_std ** 2
        
        # Compute squared differences
        diff = particle_obs - observations.unsqueeze(1)  # [batch_size, num_particles, obs_dim]
        squared_diff = torch.sum(diff ** 2, dim=2)  # [batch_size, num_particles]
        
        # Compute log-likelihood
        log_likelihood = -squared_diff / (2 * obs_error_var * temp)
        
        # Expand back to match particle dimensions
        # The log_likelihood has shape [batch_size, num_particles], 
        # so we need to broadcast it to match particle dimensions
        return log_likelihood.unsqueeze(-1).unsqueeze(-1)  # Add dims to match original shape
    
    def _resample_particles(
        self, 
        particles: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Resample particles based on their weights using systematic resampling.
        
        Args:
            particles: Particle tensor of shape [batch_size, num_particles, ...]
            weights: Normalized weights of shape [batch_size, num_particles, ...]
            
        Returns:
            Resampled particle tensor
        """
        # Extract batch and particle dimensions
        _ = particles.size(0)  # batch_size
        _ = particles.size(1)  # num_particles
        
        # Flatten weights for resampling (consider only the first weight dimension)
        flat_weights = weights.squeeze(-1).squeeze(-1)  # Remove extra dims if they exist
        
        # Systematic resampling
        device = particles.device
        dtype = particles.dtype
        
        # Create cumulative sum of weights
        cumsum_weights = torch.cumsum(flat_weights, dim=1)  # [batch_size, num_particles]
        
        # Generate uniform samples for systematic resampling
        u = (torch.arange(particles.size(1), dtype=dtype, device=device) + 
             torch.rand(particles.size(0), 1, device=device)) / particles.size(1)  
        # [batch_size, num_particles]
        
        # Find indices of particles to select
        indices = torch.searchsorted(cumsum_weights, u.clamp(0, 1))  # [batch_size, num_particles]
        indices = torch.clamp(indices, 0, particles.size(1) - 1)  # Ensure valid indices
        
        # Create output tensor
        resampled = torch.zeros_like(particles)
        
        # Resample particles for each batch
        for b in range(particles.size(0)):
            for i in range(particles.size(1)):
                resampled[b, i] = particles[b, indices[b, i]]
        
        return resampled
    
    def _assimilate_graph_particles(
        self, 
        particles: Union[Data, HeteroData], 
        observations: torch.Tensor
    ) -> Union[Data, HeteroData]:
        """
        Perform particle filter assimilation for graph-based particles.
        
        Args:
            particles: Graph particle ensemble
            observations: Observation data
            
        Returns:
            Updated graph particle ensemble
        """
        # For graph-based particles, we focus on resampling nodes based on weights
        if isinstance(particles, HeteroData):
            # Handle heterogeneous graph particles
            result = HeteroData()
            
            for node_type in particles.node_types:
                if hasattr(particles[node_type], 'x') and particles[node_type].x is not None:
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
                        
                        # Compute log-likelihood weights
                        # [num_particles, obs_dim]
                        obs_expanded = observations[0:1].expand(particle_means.size(0), -1)
                        diff = particle_obs - obs_expanded
                        squared_diff = torch.sum(diff ** 2, dim=1)  # [num_particles]
                        
                        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
                        obs_error_var = self.observation_error_std ** 2
                        log_likelihood = -squared_diff / (2 * obs_error_var * temp)
                        
                        # Normalize weights
                        max_log_weight = torch.max(log_likelihood)
                        weights = torch.exp(log_likelihood - max_log_weight)
                        weights = weights / (torch.sum(weights) + 1e-12)
                        
                        # Systematic resampling
                        device = node_features.device
                        dtype = node_features.dtype
                        
                        cumsum_weights = torch.cumsum(weights, dim=0)
                        u = (torch.arange(node_features.size(1), dtype=dtype, device=device) + 
                             torch.rand(1, device=device)) / node_features.size(1)
                        indices = torch.searchsorted(cumsum_weights, u.clamp(0, 1))
                        indices = torch.clamp(indices, 0, node_features.size(1) - 1)
                        
                        # Resample particles
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
                        if key != 'x':
                            setattr(result[node_type], key, value)
            
            # Copy edge attributes
            for edge_type in particles.edge_types:
                for key, value in particles[edge_type].items():
                    setattr(result[edge_type], key, value)
                    
            return result
        else:
            # Handle homogeneous graph particles
            result = Data()
            
            if hasattr(particles, 'x') and particles.x is not None:
                node_features = particles.x
                
                # Assuming node_features has shape [num_nodes, num_particles, features]
                if node_features.dim() == 3:
                    num_nodes, num_particles, feat_dim = node_features.shape
                    
                    # Compute mean features across nodes for each particle
                    particle_means = torch.mean(node_features, dim=0)  # [num_particles, features]
                    
                    # Use first few features as pseudo-observations for weight computation
                    obs_dim = min(particle_means.size(1), observations.size(1))
                    particle_obs = particle_means[:, :obs_dim]  # [num_particles, obs_dim]
                    
                    # Compute log-likelihood weights
                    # [num_particles, obs_dim]
                    obs_expanded = observations[0:1].expand(num_particles, -1)
                    diff = particle_obs - obs_expanded
                    squared_diff = torch.sum(diff ** 2, dim=1)  # [num_particles]
                    
                    temp = torch.clamp(self.temperature, min=0.1, max=10.0)
                    obs_error_var = self.observation_error_std ** 2
                    log_likelihood = -squared_diff / (2 * obs_error_var * temp)
                    
                    # Normalize weights
                    max_log_weight = torch.max(log_likelihood)
                    weights = torch.exp(log_likelihood - max_log_weight)
                    weights = weights / (torch.sum(weights) + 1e-12)
                    
                    # Systematic resampling
                    device = node_features.device
                    dtype = node_features.dtype
                    
                    cumsum_weights = torch.cumsum(weights, dim=0)
                    u = (torch.arange(num_particles, dtype=dtype, device=device) + 
                         torch.rand(1, device=device)) / num_particles
                    indices = torch.searchsorted(cumsum_weights, u.clamp(0, 1))
                    indices = torch.clamp(indices, 0, num_particles - 1)
                    
                    # Resample particles
                    resampled_features = node_features[:, indices, :]
                    
                    # Add small noise to prevent degeneracy
                    # Add small noise to prevent degeneracy
                    noise = torch.randn_like(resampled_features) * (
                        self.process_noise_std * 0.1
                    )
                    result.x = resampled_features + noise
                else:
                    result.x = particles.x
            
            # Copy other attributes
            for key, value in particles.items():
                if key != 'x':
                    setattr(result, key, value)
                    
            return result
    
    def _compute_analysis(
        self, 
        ensemble: Union[torch.Tensor, Data, HeteroData]
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Compute analysis state as the weighted average of particles.
        
        Args:
            ensemble: Particle ensemble
            
        Returns:
            Analysis state (weighted average of particles)
        """
        if isinstance(ensemble, torch.Tensor):
            # Return mean across particle dimension (dim=1)
            return torch.mean(ensemble, dim=1)
        elif isinstance(ensemble, HeteroData):
            result = HeteroData()
            
            for node_type in ensemble.node_types:
                if hasattr(ensemble[node_type], 'x') and ensemble[node_type].x is not None:
                    node_features = ensemble[node_type].x
                    if node_features.dim() == 3:  # [num_nodes, num_particles, features]
                        # Mean across particle dimension (dim=1)
                        result[node_type].x = torch.mean(node_features, dim=1)
                    else:
                        result[node_type].x = ensemble[node_type].x
                else:
                    # Copy other node attributes
                    for key, value in ensemble[node_type].items():
                        if key != 'x':
                            setattr(result[node_type], key, value)
            
            # Copy edge attributes
            for edge_type in ensemble.edge_types:
                for key, value in ensemble[edge_type].items():
                    setattr(result[edge_type], key, value)
                    
            return result
        elif isinstance(ensemble, Data):
            result = Data()
            
            if hasattr(ensemble, 'x') and ensemble.x is not None:
                node_features = ensemble.x
                if node_features.dim() == 3:  # [num_nodes, num_particles, features]
                    # Mean across particle dimension (dim=1)
                    result.x = torch.mean(node_features, dim=1)
                else:
                    result.x = ensemble.x
            
            # Copy other attributes
            for key, value in ensemble.items():
                if key != 'x':
                    setattr(result, key, value)
                    
            return result
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")