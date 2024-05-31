"""Noise generation utils."""

import numpy as np
import pyshtools as pysh


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
    return grid


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
