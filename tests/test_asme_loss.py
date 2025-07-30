import pytest
import torch
import torch_harmonics as th

from graph_weather.models.losses import AMSENormalizedLoss

# This file contains tests for the AMSENormalizedLoss class, which computes the


@pytest.fixture
def default_shape() -> tuple[int, int, int, int]:
    """Return a default tensor shape (B, C, H, W) for test inputs."""
    return 2, 3, 32, 64


@pytest.fixture
def feature_variance(default_shape: tuple) -> torch.Tensor:
    """Return a synthetic feature variance tensor, one value per channel."""
    _, num_channels, _, _ = default_shape
    return (torch.rand(num_channels) + 0.5).clone().detach()


@pytest.fixture
def loss_fn(feature_variance: torch.Tensor) -> AMSENormalizedLoss:
    """Instantiate the AMSENormalizedLoss with mock feature variance."""
    return AMSENormalizedLoss(feature_variance=feature_variance)


def test_zero_loss_for_identical_inputs(loss_fn: AMSENormalizedLoss, default_shape: tuple):
    """Loss should be zero when prediction and target tensors are identical."""
    pred = torch.randn(default_shape)
    target = pred.clone()
    loss = loss_fn(pred, target)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_positive_loss_for_different_inputs(loss_fn: AMSENormalizedLoss, default_shape: tuple):
    """Loss should be strictly positive when inputs differ."""
    pred = torch.randn(default_shape)
    target = torch.randn(default_shape)
    loss = loss_fn(pred, target)
    assert loss.item() > 0.0


def test_gradient_flow(loss_fn: AMSENormalizedLoss, default_shape: tuple):
    """Check that gradients can flow through the loss for backpropagation."""
    pred = torch.randn(default_shape, requires_grad=True)
    target = torch.randn(default_shape)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.sum(torch.abs(pred.grad)) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution(feature_variance: torch.Tensor, default_shape: tuple):
    """Verify that the loss runs on GPU and returns a finite CUDA tensor."""
    device = torch.device("cuda")
    loss_fn_cuda = AMSENormalizedLoss(feature_variance=feature_variance).to(device)
    pred = torch.randn(default_shape, device=device)
    target = torch.randn(default_shape, device=device)
    loss = loss_fn_cuda(pred, target)
    assert loss.is_cuda
    assert torch.isfinite(loss)


def test_known_value_simple_case(feature_variance: torch.Tensor):
    """
    Validate loss against a known spectral case.

    This test generates synthetic spectral coefficients and applies the inverse
    spherical harmonic transform to ensure the AMSE loss produces expected values.
    """
    nlat, nlon = 16, 32
    batch_size, num_channels = 1, feature_variance.shape[0]

    sht_forward_temp = th.RealSHT(nlat, nlon, grid="equiangular")
    lmax, mmax = sht_forward_temp.lmax, sht_forward_temp.mmax
    coeffs_shape = (batch_size * num_channels, lmax, mmax)

    # Place known energy in (l=1, m=0) band
    target_coeffs = torch.zeros(coeffs_shape, dtype=torch.complex64)
    target_coeffs[:, 1, 0] = 1.0 + 0.0j
    pred_coeffs = target_coeffs * 0.5

    # Inverse SHT to get spatial-domain data
    isht = th.InverseRealSHT(nlat, nlon, grid="equiangular")
    target = isht(target_coeffs).view(batch_size, num_channels, nlat, nlon)
    pred = isht(pred_coeffs).view(batch_size, num_channels, nlat, nlon)

    # Manually compute expected normalized spectral loss
    psd_target_l1 = 1.0**2
    psd_pred_l1 = 0.5**2
    amp_error_l1 = (
        torch.sqrt(torch.tensor(psd_pred_l1)) - torch.sqrt(torch.tensor(psd_target_l1))
    ) ** 2
    expected_spectral_loss_per_channel = amp_error_l1
    expected_normalized_loss = (expected_spectral_loss_per_channel / feature_variance).mean()

    # Compare to actual loss
    loss_fn = AMSENormalizedLoss(feature_variance=feature_variance)
    actual_loss = loss_fn(pred, target)

    assert torch.allclose(actual_loss, expected_normalized_loss, atol=1e-5)
