"""Weather loss functions"""
import numpy as np

import torch
from torch import nn
from graph_weather.utils.logger import get_logger

LOGGER = get_logger(__name__)


class NormalizedMSELoss(nn.Module):
    """Loss function described in the paper"""

    def __init__(self, feature_variance: np.ndarray, lat_lons: np.ndarray) -> None:
        """
        Normalized MSE Loss as described in the paper.

        This re-scales each physical variable such that it has unit-variance in the 3 hour temporal
        difference. E.g. for temperature data, divide every one at all pressure levels by
        sigma_t_3hr, where sigma^2_T,3hr is the variance of the 3 hour change in temperature,
        averaged across space (lat/lon + pressure levels) and time (100 random temporal frames).

        Additionally weights by the cos(lat) of the feature.

        Args:
            feature_variance: variance for each of the physical features
            lat_lons: array of lat/lon, used to generate weighting
            device: device on which the variances and weights tensors are created

        # TODO Rescale by nominal static air density at each pressure level
        """
        super().__init__()

        weights = np.cos(lat_lons[:, 0] * np.pi / 180.0) + 1.0e-4  # get rid of some small negative weight values
        LOGGER.debug(f"min/max cos(lat) weights: {weights.min():.3e}, {weights.max():.3e}")

        self.register_buffer("weights", torch.as_tensor(weights), persistent=True)
        self.register_buffer("feature_variance", torch.as_tensor(feature_variance), persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        pred = pred / self.feature_variance
        target = target / self.feature_variance

        # Mean of the physical variables
        out = torch.square(pred - target).mean(dim=-1)
        # Weight by the latitude, as that changes, so does the size of the pixel
        out = out * self.weights.expand_as(out)
        return out.mean()
