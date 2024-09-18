"""Noise generation utils."""

import einops
import numpy as np
import torch
import torch_harmonics as th


def generate_isotropic_noise(num_lon: int, num_lat: int, num_samples=1, isotropic=True):
    """Generate noise on the grid.

    When isotropic is True it samples the equivalent of white noise on a sphere and project it onto
    a grid using Driscoll and Healy, 1994, algorithm. The power spectrum is normalized to have
    variance 1. We need to assume lons = 2 * lats or lons = 2 * (lats -1). If isotropic is false, it
    samples flat normal random noise.

    Args:
        num_lon (int): number of longitudes in the grid.
        num_lat (int): number of latitudes in the grid.
        num_samples (int): number of indipendent samples. Defaults to 1.
        isotropic (bool): if true generates isotropic noise, else flat noise. Defaults to True.

    Returns:
        grid: Numpy array with shape shape(grid) x num_samples.
    """
    if isotropic:
        if 2 * num_lat == num_lon:
            extend = False
        elif 2 * (num_lat - 1) == num_lon:
            extend = True
        else:
            raise ValueError(
                "Isotropic noise requires grid's shape to be 2N x N or 2N x (N+1): "
                f"got {num_lon} x {num_lat}. If the shape is correct, please specify"
                "isotropic=False in the constructor.",
            )

    if isotropic:
        lmax = num_lat - 1 if extend else num_lat
        mmax = lmax + 1
        coeffs = torch.randn(num_samples, lmax, mmax, dtype=torch.complex64) / np.sqrt(
            (num_lat**2) // 2
        )
        isht = th.InverseRealSHT(
            nlat=num_lat, nlon=num_lon, lmax=lmax, mmax=mmax, grid="equiangular"
        )
        noise = isht(coeffs) * np.sqrt(2 * np.pi)
        noise = einops.rearrange(noise, "b lat lon -> lon lat b").numpy()
    else:
        noise = np.random.randn(num_lon, num_lat, num_samples)
    return noise


def sample_noise_level(sigma_min=0.02, sigma_max=88, rho=7):
    """Generate random sample of noise level.

    Sample a noise level according to the distribution described in the paper.
    Notice that the default values are valid only for training and need to be
    modified for sampling.

    Args:
        sigma_min (float, optional): Defaults to 0.02.
        sigma_max (int, optional): Defaults to 88.
        rho (int, optional): Defaults to 7.

    Returns:
        noise_level: single sample of noise level.
    """
    u = np.random.random()
    noise_level = (
        sigma_max ** (1 / rho) + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return noise_level


class Preconditioner(torch.nn.Module):
    """Collection of preconditioning functions.

    These functions are described in Karras (2022), table 1.
    """

    def __init__(self, sigma_data: float = 1):
        """Initialize the preconditioning functions.

        Args:
            sigma_data (float): Karras suggests 0.5, GenCast 1. Defaults to 1.
        """
        super().__init__()
        self.sigma_data = sigma_data

    def c_skip(self, sigma):
        """Scaling factor for skip connection."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Scaling factor for output."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma):
        """Scaling factor for input."""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        """Scaling factor for noise level."""
        return 1 / 4 * torch.log(sigma)
