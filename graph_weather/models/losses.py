"""Weather loss functions"""
import numpy as np
import torch


class NormalizedMSELoss(torch.nn.Module):
    """Loss function described in the paper"""

    def __init__(self, feature_variance: list, lat_lons: list, device="cpu"):
        """
        Normalized MSE Loss as described in the paper

        This re-scales each physical variable such that it has unit-variance in the 3 hour temporal
        difference. E.g. for temperature data, divide every one at all pressure levels by
        sigma_t_3hr, where sigma^2_T,3hr is the variance of the 3 hour change in temperature,
         averaged across space (lat/lon + pressure levels) and time (100 random temporal frames).

         Additionally weights by the cos(lat) of the feature

        Args:
            feature_variance: Variance for each of the physical features
            lat_lons: List of lat/lon pairs, used to generate weighting
        """
        # TODO Rescale by nominal static air density at each pressure level
        super().__init__()
        self.feature_variance = torch.tensor(feature_variance)
        weights = []
        for lat, lon in lat_lons:
            weights.append(np.cos(lat))
        self.weights = torch.tensor(weights, dtype=torch.float)

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

        pred = pred / self.feature_variance
        target = target / self.feature_variance

        out = (pred - target) ** 2
        # Mean of the physical variables
        out = out.mean(-1)
        # Weight by the latitude, as that changes, so does the size of the pixel
        out = out * self.weights.expand_as(out)
        return out.mean()
