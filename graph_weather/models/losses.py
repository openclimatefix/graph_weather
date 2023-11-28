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

         cos and sin should be in radians

        Args:
            feature_variance: Variance for each of the physical features
            lat_lons: List of lat/lon pairs, used to generate weighting
        """
        # TODO Rescale by nominal static air density at each pressure level
        super().__init__()
        self.feature_variance = torch.tensor(feature_variance)
        assert not torch.isnan(self.feature_variance).any()
        weights = []
        for lat, lon in lat_lons:
            weights.append(np.cos(lat * np.pi / 180.0))
        self.weights = torch.tensor(weights, dtype=torch.float)
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

        # pred = pred / self.feature_variance
        # target = target / self.feature_variance

        out = (pred - target) ** 2
        assert not torch.isnan(out).any()
        # Mean of the physical variables
        out = out.mean(-1)
        # Weight by the latitude, as that changes, so does the size of the pixel
        out = out * self.weights.expand_as(out)
        assert not torch.isnan(out).any()
        return out.mean()


# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


class CellAreaWeightedLossFunction(nn.Module):
    """Loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """

        loss = (invar - outvar) ** 2
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss


class CustomCellAreaWeightedLossAutogradFunction(torch.autograd.Function):
    """Autograd fuunction for custom loss with cell area weighting."""

    @staticmethod
    def forward(ctx, invar: torch.Tensor, outvar: torch.Tensor, area: torch.Tensor):
        """Forward of custom loss function with cell area weighting."""

        diff = invar - outvar  # T x C x H x W
        loss = diff**2
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, area)
        loss = loss.mean()
        loss_grad = diff * (2.0 / (math.prod(invar.shape)))
        loss_grad *= area.unsqueeze(0).unsqueeze(0)
        ctx.save_for_backward(loss_grad)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: torch.Tensor):
        """Backward method of custom loss function with cell area weighting."""

        # grad_loss should be 1, multiply nevertheless
        # to avoid issues with cases where this isn't the case
        (grad_invar,) = ctx.saved_tensors
        return grad_invar * grad_loss, None, None


class CustomCellAreaWeightedLossFunction(CellAreaWeightedLossFunction):
    """Custom loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area: torch.Tensor):
        super().__init__(area)

    def forward(self, invar: torch.Tensor, outvar: torch.Tensor) -> torch.Tensor:
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """

        return CustomCellAreaWeightedLossAutogradFunction.apply(invar, outvar, self.area)
