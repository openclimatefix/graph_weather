import torch
import numpy as np

"""

For the loss, paper uses MSE loss, but after some processing steps:

Re-scale each physical variable such that it has unit-variance in 3 hour temporal difference.
i.e. divide temperature data at all pressure levels by sigma_t_3hr, where sigma^2_T,3hr is the variance of the 3 hour change
in temperature, averaged across space (lat/lon + pressure levels) and time (100 random temporal frames).
Motivations: 1. interested in dynamics of system, so normalizing by magnitude of dynamics is appropriate
2. Physicallymeaningful unit of error should count the same whether it is happening at lower or upper levels of the atmosphere

They also rescale by nominal static air density at each pressure level, but did not have strong impact on performance.

When summing loss across the lat/lon grid, use a weight proportional to each pixels area i.e. cos(lat) weighting

So the loss has to do a few things:

1. Rescale physical variables to unit variance in 3 hour temporal difference, averaged across space and time
Calculate 1. first beforehand with the sets of variables, then use that unit variance which is the same for all



"""


class NormalizedMSELoss(torch.nn.Module):
    def __init__(self, feature_variance: list, lat_lons: list):
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
        super().__init__()
        self.feature_variance = torch.tensor(feature_variance)
        weights = []
        for lat, lon in lat_lons:
            weights.append(np.cos(lat))
        self.weights = torch.from_numpy(weights)

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

        pred = pred / self.feature_variance
        target = target / self.feature_variance

        out = (pred - target)**2
        # Weight by the latitude, as that changes, so does the size of the pixel
        out = out * self.weights.expand_as(out)
        return out.sum(0)
