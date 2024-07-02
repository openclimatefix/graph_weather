"""The weighted loss function for GenCast training."""

from typing import Optional

import numpy as np
import torch


class WeightedMSELoss(torch.nn.Module):
    """Module WeightedMSELoss.

    This module implement the loss described in GenCast's paper.
    """

    def __init__(
        self,
        grid_lat: Optional[torch.Tensor] = None,
        pressure_levels: Optional[torch.Tensor] = None,
        num_atmospheric_features: Optional[int] = None,
        single_features_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize the WeightedMSELoss Module.

        More details about the features weights are reported in GraphCast's paper. In short, if the
        single features are "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure" and "total_precipitation_12hr", then it's suggested to set
        corresponding weights as 1, 0.1, 0.1, 0.1 and 0.1.

        Args:
            grid_lat (torch.Tensor, optional): 1D tensor containing all the latitudes.
            pressure_levels (torch.Tensor, optional): 1D tensor containing all the pressure levels
                per variable.
            num_atmospheric_features (int, optional): number of atmospheric features.
            single_features_weights (torch.Tensor, optional): 1D tensor containing single features
                weights.
        """
        super().__init__()

        self.area_weights = None
        self.features_weights = None

        if grid_lat is not None:
            self.area_weights = torch.cos(grid_lat * np.pi / 180.0)

        if (
            pressure_levels is not None
            and num_atmospheric_features is not None
            and single_features_weights is not None
        ):
            pressure_weights = pressure_levels / torch.sum(pressure_levels)
            self.features_weights = torch.cat(
                (pressure_weights.repeat(num_atmospheric_features), single_features_weights), dim=-1
            )
        elif (
            pressure_levels is not None
            or num_atmospheric_features is not None
            or single_features_weights is not None
        ):
            raise ValueError(
                "Please to use features weights provide all three: pressure_levels,"
                "num_atmospheric_features and single_features_weights."
            )

        self.sigma_data = 1  # assuming normalized data!

    def _lambda_sigma(self, noise_level):
        noise_weights = (noise_level**2 + self.sigma_data**2) / (noise_level * self.sigma_data) ** 2
        return noise_weights  # [batch, 1]

    def forward(
        self,
        pred: torch.Tensor,
        noise_level: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred (torch.Tensor): prediction of the model [batch, lon, lat, var].
            noise_level (torch.Tensor): noise levels fed to the model for the corresponding
                predictions [batch, 1].
            target (torch.Tensor): target tensor [batch, lon, lat, var].

        Returns:
            torch.Tensor: weighted MSE loss.
        """
        # check shapes
        if not (pred.shape == target.shape):
            raise ValueError(
                "Predictions and targets must have same shape. The actual shapes "
                f"are {pred.shape} and {target.shape}."
            )
        if not (len(pred.shape) == 4):
            raise ValueError(
                "The expected shape for predictions and targets is "
                f"[batch, lon, lat, var], but got {pred.shape}."
            )
        if not (noise_level.shape == (pred.shape[0], 1)):
            raise ValueError(
                f"The expected shape for noise levels is [batch, 1], but got {noise_level.shape}."
            )

        # compute square residuals
        loss = (pred - target) ** 2  # [batch, lon, lat, var]
        if torch.isnan(loss).any():
            raise ValueError("NaN values encountered in loss calculation.")

        # apply area and features weights to residuals
        if self.area_weights is not None:
            if not (len(self.area_weights) == pred.shape[2]):
                raise ValueError(
                    f"The size of grid_lat at initialization ({len(self.area_weights)}) "
                    f"and the number of latitudes in predictions ({pred.shape[2]}) "
                    "don't match."
                )
            loss *= self.area_weights[None, None, :, None]

        if self.features_weights is not None:
            if not (len(self.features_weights) == pred.shape[-1]):
                raise ValueError(
                    f"The size of features weights at initialization ({len(self.features_weights)})"
                    f" and the number of features in predictions ({pred.shape[-1]}) "
                    "don't match."
                )
            loss *= self.features_weights[None, None, None, :]

        # compute means across lon, lat, var for each sample in the batch
        loss = loss.flatten(1).mean(-1)  # [batch]

        # weight each sample using the corresponding noise level, then return the mean.
        loss *= self._lambda_sigma(noise_level).flatten()
        return loss.mean()
