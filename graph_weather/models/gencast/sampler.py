"""Diffusion sampler"""

import math

import torch

from graph_weather.models.gencast import Denoiser
from graph_weather.models.gencast.utils.noise import generate_isotropic_noise


class Sampler:
    """Sampler for the denoiser.

    The sampler consists in the second-order DPMSolver++2S solver (Lu et al., 2022), augmented with
    the stochastic churn (again making use of the isotropic noise) and noise inflation techniques
    used in Karras et al. (2022) to inject further stochasticity into the sampling process. In
    conditioning on previous timesteps it follows the Conditional Denoising Estimator approach
    outlined and motivated by Batzolis et al. (2021).
    """

    def __init__(
        self,
        S_noise: float = 1.05,
        S_tmin: float = 0.75,
        S_tmax: float = 80.0,
        S_churn: float = 2.5,
        r: float = 0.5,
        sigma_max: float = 80.0,
        sigma_min: float = 0.03,
        rho: float = 7,
        num_steps: int = 20,
    ):
        """Initialize the sampler.

        Args:
            S_noise (float): noise inflation parameter. Defaults to 1.05.
            S_tmin (float): minimum noise for sampling. Defaults to 0.75.
            S_tmax (float): maximum noise for sampling. Defaults to 80.
            S_churn (float): stochastic churn rate. Defaults to 2.5.
            r (float): _description_. Defaults to 0.5.
            sigma_max (float): maximum value of sigma for sigma's distribution. Defaults to 80.
            sigma_min (float): minimum value of sigma for sigma's distribution. Defaults to 0.03.
            rho (float): exponent of the sigma's distribution. Defaults to 7.
            num_steps (int): number of timesteps during sampling. Defaults to 20.
        """
        self.S_noise = S_noise
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_churn = S_churn
        self.r = r
        self.num_steps = num_steps

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho

    def _sigmas_fn(self, u):
        return (
            self.sigma_max ** (1 / self.rho)
            + u * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho

    @torch.no_grad()
    def sample(self, denoiser: Denoiser, prev_inputs: torch.Tensor):
        """Generate a sample from random noise for the given inputs.

        Args:
            denoiser (Denoiser): the denoiser model.
            prev_inputs (torch.Tensor): previous two timesteps.

        Returns:
            torch.Tensor: normalized residuals predicted.
        """
        device = prev_inputs.device

        time_steps = torch.arange(0, self.num_steps).to(device) / (self.num_steps - 1)
        sigmas = self._sigmas_fn(time_steps)

        batch_ones = torch.ones(1, 1).to(device)

        # initialize noise
        x = sigmas[0] * torch.tensor(
            generate_isotropic_noise(
                num_lon=denoiser.num_lon,
                num_lat=denoiser.num_lat,
                num_samples=denoiser.output_features_dim,
            )
        ).unsqueeze(0).to(device)

        for i in range(len(sigmas) - 1):
            # stochastic churn from Karras et al. (Alg. 2)
            gamma = (
                min(self.S_churn / self.num_steps, math.sqrt(2) - 1)
                if self.S_tmin <= sigmas[i] <= self.S_tmax
                else 0.0
            )
            # noise inflation from Karras et al. (Alg. 2)
            noise = self.S_noise * torch.tensor(
                generate_isotropic_noise(
                    num_lon=denoiser.num_lon,
                    num_lat=denoiser.num_lat,
                    num_samples=denoiser.output_features_dim,
                )
            )
            noise = noise.unsqueeze(0).to(device)

            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigmas[i] ** 2) ** 0.5 * noise
            denoised = denoiser(x, prev_inputs, sigma_hat * batch_ones)

            if i == len(sigmas) - 2:
                # final Euler step
                d = (x - denoised) / sigma_hat
                x = x + d * (sigmas[i + 1] - sigma_hat)
            else:
                # DPMSolver++2S  step (Alg. 1 in Lu et al.) with alpha_t=1.
                # t_{i-1} is t_hat because of stochastic churn!
                lambda_hat = -torch.log(sigma_hat)
                lambda_next = -torch.log(sigmas[i + 1])
                h = lambda_next - lambda_hat
                lambda_mid = lambda_hat + self.r * h
                sigma_mid = torch.exp(-lambda_mid)

                u = sigma_mid / sigma_hat * x - (torch.exp(-self.r * h) - 1) * denoised
                denoised_2 = denoiser(u, prev_inputs, sigma_mid * batch_ones)
                D = (1 - 1 / (2 * self.r)) * denoised + 1 / (2 * self.r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x
