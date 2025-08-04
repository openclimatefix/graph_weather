"""Weather loss functions"""

import numpy as np
import torch
import torch.nn as nn
import torch_harmonics as th


class NormalizedMSELoss(torch.nn.Module):
    """Loss function described in the paper"""

    def __init__(
        self, feature_variance: list, lat_lons: list, device="cpu", normalize: bool = False
    ):
        """
        Normalized MSE Loss as described in the paper

        This re-scales each physical variable such that it has unit-variance in the 3 hour temporal
        difference. E.g. for temperature data, divide every one at all pressure levels by
        sigma_t_3hr, where sigma^2_T,3hr is the variance of the 3 hour change in temperature,
         averaged across space (lat/lon + pressure levels) and time (100 random temporal frames).

         Additionally weights by the cos(lat) of the feature

         cos and sin should be in radians

        Args:
            feature_variance: Variance for each of the physical features
            lat_lons: List of lat/lon pairs, used to generate weighting
            device: checks for device whether it supports gpu or not
            normalize: option for normalize
        """
        # TODO Rescale by nominal static air density at each pressure level, could be 1/pressure level or something similar
        super().__init__()
        self.feature_variance = torch.tensor(feature_variance)
        assert not torch.isnan(self.feature_variance).any()
        # Compute unique latitudes from the provided lat/lon pairs.
        unique_lats = sorted(set(lat for lat, _ in lat_lons))
        # Use the cosine of each unique latitude (converted to radians) as its weight.
        self.weights = torch.tensor(
            [np.cos(lat * np.pi / 180.0) for lat in unique_lats], dtype=torch.float
        )
        self.normalize = normalize
        assert not torch.isnan(self.weights).any()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the loss

        Rescales both predictions and target, so assumes neither are already normalized
        Additionally weights by the cos(lat) of the set of features

        Args:
            pred: Prediction tensor
            target: Target tensor

        Returns:
            MSE loss on the variance-normalized values
        """
        self.feature_variance = self.feature_variance.to(pred.device)
        self.weights = self.weights.to(pred.device)
        print(pred.shape)
        print(target.shape)
        print(self.weights.shape)

        out = (pred - target) ** 2
        print(out.shape)
        if self.normalize:
            out = out / self.feature_variance

        assert not torch.isnan(out).any()
        # Mean of the physical variables
        out = out.mean(-1)

        # Flatten all dimensions except the batch dimension.
        B, *dims = out.shape
        num_nodes = np.prod(
            dims
        )  # Total number of grid nodes (e.g., if grid is HxW, then num_nodes = H*W)
        out = out.view(B, num_nodes)

        # Determine the number of unique latitude weights and infer the number of grid columns.
        num_unique = self.weights.shape[0]  # e.g., number of unique latitudes (rows)
        num_lon = num_nodes // num_unique  # e.g. if 2592 nodes and 36 unique lat, then num_lon=72

        # Tile the unique latitude weights into a full weight grid
        weight_grid = self.weights.unsqueeze(1).expand(num_unique, num_lon).reshape(1, num_nodes)
        weight_grid = weight_grid.expand(B, num_nodes)  # Now weight_grid is [B, num_nodes]

        # Multiply the per-node error by the corresponding weight.
        out = out * weight_grid

        assert not torch.isnan(out).any()
        return out.mean()


# Spectrally Adjusted Mean Squared Error (AMSE) loss
class AMSENormalizedLoss(nn.Module):
    """
    Spectrally Adjusted Mean Squared Error (AMSE) Loss.

    This loss function is designed to address the "double penalty" issue in spatial forecasting
    by separately penalizing amplitude and phase differences in the spectral domain.

    It applies the Spherical Harmonic Transform (SHT) to both predictions and targets,
    computes the power spectral density (PSD), and then evaluates two terms:
    1. Amplitude Error (difference in spectral amplitudes).
    2. Decorrelation Error (phase misalignment/coherence loss).

    This implementation follows the formulation in:
    "Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function"
    (ICML 2025 Poster).

    Args:
        feature_variance (list or torch.Tensor): Variance of each physical feature for normalization (length C).
        epsilon (float): Small constant for numerical stability.
    """

    def __init__(self, feature_variance: list | torch.Tensor, epsilon: float = 1e-9):
        super().__init__()
        if not isinstance(feature_variance, torch.Tensor):
            feature_variance = torch.tensor(feature_variance, dtype=torch.float32)
        else:
            feature_variance = feature_variance.clone().detach().float()

        self.register_buffer("feature_variance", feature_variance)

        # SHT cache to avoid re-initializing on every forward pass since object performs some expensive pre-computation when it's initialized. Doing this repeatedly inside the training loop can add unnecessary overhead.
        self.epsilon = epsilon
        self.sht_cache = {}

    def _get_sht(self, nlat: int, nlon: int, device: torch.device) -> th.RealSHT:
        """
        Helper to get a cached SHT object, creating it if it doesn't exist.
        This prevents re-initializing the SHT object on every forward pass.
        """
        key = (nlat, nlon, device)
        if key not in self.sht_cache:
            self.sht_cache[key] = th.RealSHT(nlat, nlon, grid="equiangular").to(device)
        return self.sht_cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the AMSE loss.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, C, H, W).
            target (torch.Tensor): Ground truth tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value (averaged over batch and features).
        """
        if pred.shape != target.shape:
            raise ValueError("Prediction and target tensors must have the same shape.")
        if pred.ndim != 4:
            raise ValueError("Input tensors must be 4D: (batch, channels, lat, lon)")

        batch_size, num_channels, nlat, nlon = pred.shape

        # Reshape to (B*C, H, W) to process all variables at once
        pred_reshaped = pred.view(batch_size * num_channels, nlat, nlon)
        target_reshaped = target.view(batch_size * num_channels, nlat, nlon)

        # Get the (potentially cached) SHT object
        sht = self._get_sht(nlat, nlon, pred.device)
        pred_coeffs = sht(pred_reshaped)  # (B*C, L, M) complex
        target_coeffs = sht(target_reshaped)  # (B*C, L, M) complex

        # Compute Power Spectral Densities (PSD): sum |coeff|^2 over M
        pred_psd = torch.sum(torch.abs(pred_coeffs) ** 2, dim=-1)  # (B*C, L)
        target_psd = torch.sum(torch.abs(target_coeffs) ** 2, dim=-1)  # (B*C, L)

        # Compute spectral coherence between prediction and target
        cross_power = pred_coeffs * torch.conj(target_coeffs)  # (B*C, L, M)
        coherence_num = torch.sum(cross_power.real, dim=-1)  # (B*C, L)
        coherence_denom = torch.sqrt(pred_psd * target_psd)
        coherence = coherence_num / (coherence_denom + self.epsilon)  # (B*C, L)

        # Compute amplitude error: difference in sqrt(PSD)
        amp_error = (
            torch.sqrt(pred_psd + self.epsilon) - torch.sqrt(target_psd + self.epsilon)
        ) ** 2

        # Compute decorrelation error
        decor_error = 2.0 * coherence_denom * (1.0 - coherence)

        # Total spectral loss per sample
        spectral_loss = torch.sum(amp_error + decor_error, dim=-1)  # (B*C,)

        # Reshape back to (B, C)
        spectral_loss = spectral_loss.view(batch_size, num_channels)

        # Normalize by feature-wise variance and compute mean loss
        normalized_loss = spectral_loss / (self.feature_variance + self.epsilon)
        return normalized_loss.mean()
