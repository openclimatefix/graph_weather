import pytest
import torch
import torch_harmonics as th

from graph_weather.models.losses import AMSENormalizedLoss


@pytest.fixture
def default_shape():
    """Provides a default shape for tensors: (B, C, H, W)."""
    return (2, 3, 32, 64)


@pytest.fixture
def feature_variance(default_shape):
    """Provides a default feature variance tensor."""
    _, num_channels, _, _ = default_shape
    return (torch.rand(num_channels) + 0.5).clone().detach()


@pytest.fixture
def loss_fn(feature_variance):
    """Provides an instance of the AMSENormalizedLoss."""
    return AMSENormalizedLoss(feature_variance=feature_variance)


class TestAMSENormalizedLoss:
    """Test suite for the AMSENormalizedLoss class."""

    def test_zero_loss_for_identical_inputs(self, loss_fn, default_shape):
        pred = torch.randn(default_shape)
        target = pred.clone()
        loss = loss_fn(pred, target)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_loss_for_different_inputs(self, loss_fn, default_shape):
        pred = torch.randn(default_shape)
        target = torch.randn(default_shape)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.0

    def test_gradient_flow(self, loss_fn, default_shape):
        pred = torch.randn(default_shape, requires_grad=True)
        target = torch.randn(default_shape)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.sum(torch.abs(pred.grad)) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, feature_variance, default_shape):
        device = torch.device("cuda")
        loss_fn_cuda = AMSENormalizedLoss(feature_variance=feature_variance).to(device)
        pred = torch.randn(default_shape, device=device)
        target = torch.randn(default_shape, device=device)
        loss = loss_fn_cuda(pred, target)
        assert loss.is_cuda
        assert torch.isfinite(loss)

    def test_known_value_simple_case(self, feature_variance):
        """
        Tests the loss function against a known value by creating a simple case
        in the spectral domain to avoid discretization errors (spectral leakage).
        """
        # Setup
        nlat, nlon = 16, 32
        batch_size, num_channels = 1, feature_variance.shape[0]

        # Get lmax and mmax from a temporary forward SHT object
        sht_forward_temp = th.RealSHT(nlat, nlon, grid="equiangular")
        lmax, mmax = sht_forward_temp.lmax, sht_forward_temp.mmax
        coeffs_shape = (batch_size * num_channels, lmax, mmax)

        # 1. Create perfect coefficients directly in the spectral domain
        target_coeffs = torch.zeros(coeffs_shape, dtype=torch.complex64)
        target_coeffs[:, 1, 0] = 1.0 + 0.0j
        pred_coeffs = target_coeffs * 0.5

        # 2. Use the INVERSE transform object to create the spatial grids
        isht = th.InverseRealSHT(nlat, nlon, grid="equiangular")
        target = isht(target_coeffs)
        pred = isht(pred_coeffs)

        # Reshape to (B, C, H, W)
        target = target.view(batch_size, num_channels, nlat, nlon)
        pred = pred.view(batch_size, num_channels, nlat, nlon)

        # 3. Calculate expected loss
        psd_target_l1 = 1.0**2
        psd_pred_l1 = 0.5**2
        amp_error_l1 = (
            torch.sqrt(torch.tensor(psd_pred_l1)) - torch.sqrt(torch.tensor(psd_target_l1))
        ) ** 2
        expected_spectral_loss_per_channel = amp_error_l1
        expected_normalized_loss = (expected_spectral_loss_per_channel / feature_variance).mean()

        # 4. Compute actual loss
        loss_fn = AMSENormalizedLoss(feature_variance=feature_variance)
        actual_loss = loss_fn(pred, target)

        # 5. Assert they are close
        assert torch.allclose(actual_loss, expected_normalized_loss, atol=1e-5)
