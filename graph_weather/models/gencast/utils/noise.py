"""Noise generation utils."""

import numpy as np
import pyshtools as pysh
import torch


def generate_isotropic_noise(num_lat, num_samples=1):
    """Generate isotropic noise on the grid.

    Sample the equivalent of white noise on a sphere and project it onto a grid using
    Driscoll and Healy, 1994 algorithm. The power spectrum is normalized to have variance 1.
    We need to assume lons = 2* lats.

    Args:
        num_lat (int): Number of latitudes in the final grid.
        num_samples (int, optional): Number of indipendent samples. Defaults to 1.

    Returns:
        grid: Numpy array with shape shape(grid) x num_samples.
    """
    power = np.ones(num_lat // 2, dtype=float) / (
        num_lat // 2
    )  # normalized to get each point with std 1
    grid = np.zeros((num_lat * 2, num_lat, num_samples))
    for i in range(num_samples):
        clm = pysh.SHCoeffs.from_random(power)
        grid[:, :, i] = clm.expand(grid="DH2", extend=False).to_array().transpose()
    return grid.astype(np.float32)


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
        return self.sigma_data / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Scaling factor for output."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma):
        """Scaling factor for input."""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        """Scaling factor for noise level."""
        return 1 / 4 * torch.log(sigma)
