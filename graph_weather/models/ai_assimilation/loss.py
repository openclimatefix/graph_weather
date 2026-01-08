

from typing import Optional, Union

import numpy as np
import torch
from torch.nn import Module


class ThreeDVarLoss(Module):
    """
    3D-Var cost function loss for self-supervised AI-based data assimilation.

    The 3D-Var cost function is defined as:
    J(x) = (x - x_b)^T B^{-1}(x - x_b) + (y - Hx)^T R^{-1}(y - Hx)

    Where:
    - x: analysis state (model output)
    - x_b: background state (first guess)
    - y: observations
    - B: background error covariance matrix
    - R: observation error covariance matrix
    - H: observation operator matrix
    """
    def __init__(
        self,
        background_error_covariance: Optional[Union[torch.Tensor, np.ndarray]] = None,
        observation_error_covariance: Optional[Union[torch.Tensor, np.ndarray]] = None,
        observation_operator: Optional[Union[torch.Tensor, np.ndarray]] = None,
        bg_weight: float = 1.0,
        obs_weight: float = 1.0,
    ):
        
        super(ThreeDVarLoss, self).__init__()

        self.bg_weight = bg_weight
        self.obs_weight = obs_weight

        # Initialize background error covariance B
        if background_error_covariance is None:
            # Default to identity matrix (diagonal with unit variance)
            self.register_buffer("B_inv", None)  # Will be computed as identity when needed
        else:
            if isinstance(background_error_covariance, torch.Tensor):
                B_inv = torch.inverse(background_error_covariance)
            else:
                B_inv = torch.inverse(
                    torch.tensor(background_error_covariance, dtype=torch.float32)
                )
            self.register_buffer("B_inv", B_inv)

        # Initialize observation error covariance R
        if observation_error_covariance is None:
            # Default to identity matrix (diagonal with unit variance)
            self.register_buffer("R_inv", None)  # Will be computed as identity when needed
        else:
            if isinstance(observation_error_covariance, torch.Tensor):
                R_inv = torch.inverse(observation_error_covariance)
            else:
                R_inv = torch.inverse(
                    torch.tensor(observation_error_covariance, dtype=torch.float32)
                )
            self.register_buffer("R_inv", R_inv)

        # Initialize observation operator H
        if observation_operator is None:
            # Default to identity (direct observation of state variables)
            self.register_buffer("H", None)  # Will be treated as identity when needed
        else:
            if isinstance(observation_operator, torch.Tensor):
                H = observation_operator
            else:
                H = torch.tensor(observation_operator, dtype=torch.float32)
            self.register_buffer("H", H)

    def forward(
        self, analysis: torch.Tensor, first_guess: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        batch_size = analysis.size(0)

        # Background term: (x - x_b)^T B^{-1} (x - x_b)
        bg_diff = analysis - first_guess

        if self.B_inv is None:
            # Use identity matrix for B^{-1} - element-wise square and sum
            bg_term = torch.sum(bg_diff * bg_diff, dim=-1)
        else:
            # Compute quadratic form (x - x_b)^T B^{-1} (x - x_b)
            # For efficiency, assuming B_inv is diagonal for now
            if self.B_inv.dim() == 1:  # Diagonal matrix stored as 1D
                bg_term = torch.sum(bg_diff * bg_diff * self.B_inv.unsqueeze(0), dim=-1)
            elif self.B_inv.dim() == 2:  # Full matrix
                bg_term = torch.sum(
                    bg_diff * torch.matmul(bg_diff.unsqueeze(-2), self.B_inv).squeeze(-2), dim=-1
                )
            else:
                raise ValueError(f"B_inv has unexpected dimensions: {self.B_inv.shape}")

        bg_term = self.bg_weight * torch.mean(bg_term)

        # Observation term: (y - Hx)^T R^{-1} (y - Hx)
        if self.H is None:
            # H is identity, so Hx = x (assuming same dimensions for obs and state)
            hx = analysis
        else:
            # Apply observation operator: Hx
            if len(analysis.shape) == 2:
                # 2D case: [batch, features]
                hx = torch.matmul(analysis, self.H.t())
            else:
                # For multi-dimensional case, we might need to reshape
                original_shape = analysis.shape
                analysis_flat = analysis.view(batch_size, -1)
                hx_flat = torch.matmul(analysis_flat, self.H.t())
                hx = hx_flat.view(original_shape)

        obs_diff = observations - hx

        if self.R_inv is None:
            # Use identity matrix for R^{-1} - element-wise square and sum
            obs_term = torch.sum(obs_diff * obs_diff, dim=-1)
        else:
            # Compute quadratic form (y - Hx)^T R^{-1} (y - Hx)
            # For efficiency, assuming R_inv is diagonal for now
            if self.R_inv.dim() == 1:  # Diagonal matrix stored as 1D
                obs_term = torch.sum(obs_diff * obs_diff * self.R_inv.unsqueeze(0), dim=-1)
            elif self.R_inv.dim() == 2:  # Full matrix
                obs_term = torch.sum(
                    obs_diff * torch.matmul(obs_diff.unsqueeze(-2), self.R_inv).squeeze(-2), dim=-1
                )
            else:
                raise ValueError(f"R_inv has unexpected dimensions: {self.R_inv.shape}")

        obs_term = self.obs_weight * torch.mean(obs_term)

        # Total 3D-Var cost
        total_loss = bg_term + obs_term

        return total_loss


class GaussianCovarianceBuilder:

    @staticmethod
    def build_gaussian_covariance(
        size: int,
        length_scale: float = 1.0,
        variance: float = 1.0,
        grid_shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        
        if grid_shape is not None and len(grid_shape) == 2:
            # For 2D grid, calculate distances based on grid positions
            h, w = grid_shape
            if h * w != size:
                raise ValueError(f"Grid shape {grid_shape} doesn't match size {size}")

            # Create coordinate grid
            coords = torch.zeros(size, 2)
            for i in range(h):
                for j in range(w):
                    idx = i * w + j
                    coords[idx, 0] = i
                    coords[idx, 1] = j

            # Calculate pairwise distances
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [size, size, 2]
            dist_sq = torch.sum(diff**2, dim=2)  # [size, size]
        else:
            # For 1D case, assume points are evenly spaced
            positions = torch.arange(size, dtype=torch.float32).unsqueeze(1)  # [size, 1]
            diff = positions - positions.t()  # [size, size]
            dist_sq = diff**2

        # Calculate Gaussian covariance: C(i,j) = variance * exp(-d^2 / (2 * length_scale^2))
        covariance = variance * torch.exp(-dist_sq / (2 * length_scale**2))

        return covariance
