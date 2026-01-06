"""
Self-Supervised Data Assimilation Framework with 3D-Var Loss

Implements a neural network that learns to produce analysis states by minimizing
the 3D-Var cost function without using ground-truth labels.
"""

import numpy as np
import torch
import torch.nn as nn


class ThreeDVarLoss(nn.Module):
    """
    Implements the 3D-Var cost function as a self-supervised loss:

    J(x) = (x - x_b)^T B^{-1} (x - x_b) + (y - Hx)^T R^{-1} (y - Hx)

    Where:
    - x: analysis state (model output)
    - x_b: background state (first guess)
    - y: observations
    - B: background error covariance
    - R: observation error covariance
    - H: observation operator
    """

    def __init__(
        self,
        background_error_covariance=None,
        observation_error_covariance=None,
        observation_operator=None,
        bg_weight=1.0,
        obs_weight=1.0,
    ):
        """Initialize the 3D-Var loss function.

        Args:
            background_error_covariance: B matrix (background error covariance)
            observation_error_covariance: R matrix (observation error covariance)
            observation_operator: H matrix (observation operator)
            bg_weight: Weight for background term
            obs_weight: Weight for observation term
        """
        super(ThreeDVarLoss, self).__init__()

        self.bg_weight = bg_weight
        self.obs_weight = obs_weight

        # Initialize background error covariance B
        if background_error_covariance is None:
            # Default to identity matrix (diagonal with unit variance)
            self.B_inv = None  # Will be computed as identity when needed
        else:
            if isinstance(background_error_covariance, torch.Tensor):
                self.B_inv = torch.inverse(background_error_covariance)
            else:
                self.B_inv = torch.inverse(torch.tensor(background_error_covariance))

        # Initialize observation error covariance R
        if observation_error_covariance is None:
            # Default to identity matrix (diagonal with unit variance)
            self.R_inv = None  # Will be computed as identity when needed
        else:
            if isinstance(observation_error_covariance, torch.Tensor):
                self.R_inv = torch.inverse(observation_error_covariance)
            else:
                self.R_inv = torch.inverse(torch.tensor(observation_error_covariance))

        # Initialize observation operator H
        if observation_operator is None:
            # Default to identity (direct observation of state variables)
            self.H = None  # Will be treated as identity when needed
        else:
            if isinstance(observation_operator, torch.Tensor):
                self.H = observation_operator
            else:
                self.H = torch.tensor(observation_operator)

    def forward(self, analysis, background, observations):
        """Compute the 3D-Var loss.

        Args:
            analysis: Model output (analysis state x)
            background: Background state (x_b)
            observations: Observations (y)

        Returns:
            Total loss value
        """
        batch_size = analysis.size(0)

        # Background term: (x - x_b)^T B^{-1} (x - x_b)
        bg_diff = analysis - background

        if self.B_inv is None:
            # Use identity matrix for B^{-1}
            bg_term = torch.sum(bg_diff * bg_diff, dim=-1)  # Element-wise square and sum
        else:
            # Compute quadratic form (x - x_b)^T B^{-1} (x - x_b)
            bg_term = torch.sum(
                bg_diff * torch.matmul(bg_diff.unsqueeze(-2), self.B_inv).squeeze(-2), dim=-1
            )

        bg_term = self.bg_weight * torch.mean(bg_term)

        # Observation term: (y - Hx)^T R^{-1} (y - Hx)
        if self.H is None:
            # H is identity, so Hx = x
            hx = analysis
        else:
            # Apply observation operator: Hx
            if len(analysis.shape) == 2:
                # 2D case: [batch, features]
                hx = torch.matmul(analysis, self.H.T)
            else:
                # For multi-dimensional case, we might need to reshape
                original_shape = analysis.shape
                analysis_flat = analysis.view(batch_size, -1)
                hx_flat = torch.matmul(analysis_flat, self.H.T)
                hx = hx_flat.view(original_shape)

        obs_diff = observations - hx

        if self.R_inv is None:
            # Use identity matrix for R^{-1}
            obs_term = torch.sum(obs_diff * obs_diff, dim=-1)  # Element-wise square and sum
        else:
            # Compute quadratic form (y - Hx)^T R^{-1} (y - Hx)
            obs_term = torch.sum(
                obs_diff * torch.matmul(obs_diff.unsqueeze(-2), self.R_inv).squeeze(-2), dim=-1
            )

        obs_term = self.obs_weight * torch.mean(obs_term)

        # Total 3D-Var cost
        total_loss = bg_term + obs_term

        return total_loss


class DataAssimilationModel(nn.Module):
    """
    Neural network model for self-supervised data assimilation.

    Takes background state and observations as input and produces an analysis state
    that minimizes the 3D-Var cost function.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.1, activation="relu"):
        """Initialize the data assimilation model.

        Args:
            input_dim: Dimension of the input state
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super(DataAssimilationModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Encoder to combine background and observations
        layers = []
        layers.append(nn.Linear(input_dim * 2, hidden_dim))  # bg + obs
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))

        # Output layer to produce analysis
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, background, observations):
        """Forward pass of the data assimilation model.

        Args:
            background: Background state (x_b)
            observations: Observations (y)

        Returns:
            analysis: Analysis state (x)
        """
        # Concatenate background and observations along the feature dimension
        combined_input = torch.cat([background, observations], dim=-1)

        # Pass through the network to get analysis
        analysis = self.network(combined_input)

        return analysis


def create_observation_operator(grid_size, obs_fraction=0.5, obs_locations=None):
    """
    Create a simple observation operator H that selects a subset of grid points

    Args:
        grid_size: Size of the grid (can be int for 1D or tuple for 2D)
        obs_fraction: Fraction of grid points that have observations
        obs_locations: Specific locations of observations (optional)

    Returns:
        H: Observation operator matrix
    """
    total_size = np.prod(grid_size) if isinstance(grid_size, (tuple, list)) else grid_size

    if obs_locations is None:
        # Randomly select observation locations
        num_obs = int(total_size * obs_fraction)
        obs_indices = np.random.choice(total_size, size=num_obs, replace=False)
    else:
        obs_indices = obs_locations
        num_obs = len(obs_indices)

    # Create H matrix (num_obs x total_size)
    H = torch.zeros(num_obs, total_size)
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1.0

    return H
