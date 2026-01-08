"""
Loss Module for AI-based Data Assimilation

Implements physics-based loss functions for training AI-based data assimilation models
without requiring ground-truth labels, following the 3D-Var approach.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ThreeDVarLoss(nn.Module):
    """
    3D-Var loss function for AI-based data assimilation.

    Implements the traditional 3D-Var cost function that balances fit to
    background (first-guess) state and observations.
    """

    def __init__(
        self,
        background_error_covariance: Optional[torch.Tensor] = None,
        observation_error_covariance: Optional[torch.Tensor] = None,
        observation_operator: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the 3D-Var loss function.

        Args:
            background_error_covariance: Background error covariance matrix B
            observation_error_covariance: Observation error covariance matrix R
            observation_operator: Observation operator matrix H
        """
        super().__init__()
        self.background_error_covariance = background_error_covariance
        self.observation_error_covariance = observation_error_covariance
        self.observation_operator = observation_operator

    def forward(
        self,
        analysis: torch.Tensor,
        background: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the 3D-Var cost function.

        Args:
            analysis: Analysis state produced by the AI model
            background: Background (first-guess) state
            observations: Observation values

        Returns:
            Total 3D-Var cost as a scalar tensor
        """
        # Background term: (x_a - x_b)^T B^{-1} (x_a - x_b)
        bg_diff = analysis - background
        if self.background_error_covariance is not None:
            # Use provided covariance matrix
            inv_bg_cov = torch.inverse(self.background_error_covariance)
            bg_quadratic = bg_diff @ inv_bg_cov * bg_diff
            bg_term = torch.sum(bg_quadratic, dim=-1)
        else:
            # Simplified: assume identity covariance (sum of squares)
            bg_term = torch.sum(bg_diff ** 2, dim=-1)

        # Observation term: (y - H x_a)^T R^{-1} (y - H x_a)
        if self.observation_operator is not None:
            # Apply observation operator
            hx = torch.matmul(
                analysis.unsqueeze(1), 
                self.observation_operator.transpose(-1, -2)
            ).squeeze(1)
        else:
            # Identity observation operator (direct comparison)
            hx = analysis

        obs_diff = observations - hx
        if self.observation_error_covariance is not None:
            # Use provided covariance matrix
            inv_obs_cov = torch.inverse(self.observation_error_covariance)
            obs_quadratic = obs_diff @ inv_obs_cov * obs_diff
            obs_term = torch.sum(obs_quadratic, dim=-1)
        else:
            # Simplified: assume identity covariance (sum of squares)
            obs_term = torch.sum(obs_diff ** 2, dim=-1)

        # Combine terms with equal weighting (can be adjusted)
        total_cost = 0.5 * (torch.mean(bg_term) + torch.mean(obs_term))

        return total_cost


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss combining 3D-Var with physical constraints.

    Extends the basic 3D-Var loss with additional physics-based regularization terms.
    """

    def __init__(
        self,
        three_d_var_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        conservation_weight: float = 0.05,
    ):
        """
        Initialize physics-informed loss.

        Args:
            three_d_var_weight: Weight for 3D-Var term
            smoothness_weight: Weight for spatial smoothness regularization
            conservation_weight: Weight for conservation law enforcement
        """
        super().__init__()
        self.three_d_var_weight = three_d_var_weight
        self.smoothness_weight = smoothness_weight
        self.conservation_weight = conservation_weight
        self.base_loss = ThreeDVarLoss()

    def forward(
        self,
        analysis: torch.Tensor,
        background: torch.Tensor,
        observations: torch.Tensor,
        grid_spacing: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute physics-informed loss with component breakdown.

        Args:
            analysis: Analysis state from AI model
            background: Background state
            observations: Observation values
            grid_spacing: Spatial grid spacing for derivative calculations

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Base 3D-Var loss
        three_d_var_loss = self.base_loss(analysis, background, observations)

        # Smoothness regularization (penalize spatial gradients)
        if analysis.dim() == 4:  # [batch, channels, height, width]
            # Compute spatial gradients
            dy = torch.abs(analysis[:, :, 1:, :] - analysis[:, :, :-1, :]).mean()
            dx = torch.abs(analysis[:, :, :, 1:] - analysis[:, :, :, :-1]).mean()
            smoothness_loss = (dy + dx) / 2.0
        else:
            # For 1D or other cases, use simple gradient approximation
            smoothness_loss = torch.mean(torch.abs(analysis[:, 1:] - analysis[:, :-1]))

        # Conservation constraint (enforce mass/energy conservation)
        conservation_loss = torch.abs(torch.mean(analysis - background))

        # Weighted combination
        total_loss = (
            self.three_d_var_weight * three_d_var_loss +
            self.smoothness_weight * smoothness_loss +
            self.conservation_weight * conservation_loss
        )

        # Return components for monitoring
        components = {
            'three_d_var': three_d_var_loss.item(),
            'smoothness': smoothness_loss.item(),
            'conservation': conservation_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, components