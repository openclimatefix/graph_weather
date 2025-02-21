"""Weather loss functions"""

import numpy as np
import torch


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
            [np.cos(lat * np.pi / 180.0) for lat in unique_lats],
            dtype=torch.float
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
        num_nodes = np.prod(dims)  # Total number of grid nodes (e.g., if grid is HxW, then num_nodes = H*W)
        out = out.view(B, num_nodes)

        # Determine the number of unique latitude weights and infer the number of grid columns.
        num_unique = self.weights.shape[0]   # e.g., number of unique latitudes (rows)
        num_lon = num_nodes // num_unique  # e.g. if 2592 nodes and 36 unique lat, then num_lon=72

        # Tile the unique latitude weights into a full weight grid
        weight_grid = self.weights.unsqueeze(1).expand(num_unique, num_lon).reshape(1, num_nodes)
        weight_grid = weight_grid.expand(B, num_nodes)  # Now weight_grid is [B, num_nodes]
        
        # Multiply the per-node error by the corresponding weight.
        out = out * weight_grid
        
        assert not torch.isnan(out).any()
        return out.mean()
