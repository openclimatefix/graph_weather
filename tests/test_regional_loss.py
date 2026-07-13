"""Tests for the region-weighted MSE loss."""

import torch

from graph_weather.models.losses import regional_weighted_mse


def test_all_region_equals_plain_mse():
    """When every point is in the region, any weight reduces to plain MSE."""
    pred = torch.randn(2, 5, 3)
    target = torch.randn(2, 5, 3)
    mask = torch.ones(5, dtype=torch.bool)

    plain = ((pred - target) ** 2).mean()
    loss = regional_weighted_mse(pred, target, mask, region_weight=7.0)

    assert torch.allclose(loss, plain)


def test_upweighting_region_raises_loss_when_error_is_in_region():
    """With error only on region points, a higher weight gives a higher loss."""
    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    target[0, 0] = 1.0  # error on the single in-region point only
    mask = torch.tensor([True, False, False, False])

    loss_low = regional_weighted_mse(pred, target, mask, region_weight=1.0)
    loss_high = regional_weighted_mse(pred, target, mask, region_weight=3.0)

    assert loss_high > loss_low


def test_region_weight_one_equals_plain_mse():
    """A weight of 1 is neutral: the loss is plain MSE for any mask."""
    pred = torch.randn(2, 6, 3)
    target = torch.randn(2, 6, 3)
    mask = torch.tensor([True, False, True, False, False, True])

    plain = ((pred - target) ** 2).mean()
    loss = regional_weighted_mse(pred, target, mask, region_weight=1.0)

    assert torch.allclose(loss, plain)


def test_zero_error_gives_zero_loss():
    """Identical prediction and target give zero loss regardless of weighting."""
    pred = torch.randn(2, 5, 3)
    mask = torch.tensor([True, False, True, False, True])

    loss = regional_weighted_mse(pred, pred.clone(), mask, region_weight=5.0)

    assert loss == 0.0
