import math

import torch

from ..config import AtmoRepConfig


class UncertaintyEstimator:
    def __init__(self, config: AtmoRepConfig):
        self.config = config

    def estimate_uncertainty(self, ensemble_predictions):
        """Estimate uncertainty from ensemble predictions"""
        uncertainties = {}

        for field, preds in ensemble_predictions.items():
            # Calculate mean and variance across ensemble dimension
            mean_pred = preds.mean(dim=0)  # [B, T, H, W]
            variance = ((preds - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)  # [B, T, H, W]

            # Calculate additional uncertainty metrics
            ensemble_spread = preds.std(dim=0)  # Standard deviation across ensemble
            entropy = self._calculate_entropy(preds)  # Information entropy

            uncertainties[field] = {
                "variance": variance,
                "spread": ensemble_spread,
                "entropy": entropy,
            }

        return uncertainties

    def _calculate_entropy(self, ensemble_preds, num_bins=10):
        """Calculate entropy by binning predictions and computing Shannon entropy"""
        # For continuous variables, we discretize into bins
        E, B, T, H, W = ensemble_preds.shape

        # Reshape for binning
        flat_preds = ensemble_preds.reshape(E, -1)

        # Find global min and max for consistent binning
        global_min = flat_preds.min()
        global_max = flat_preds.max()
        bin_width = (global_max - global_min) / num_bins

        # Initialize entropy map
        entropy = torch.zeros(B, T, H, W)

        # Calculate entropy for each pixel location
        for b in range(B):
            for t in range(T):
                for h in range(H):
                    for w in range(W):
                        # Get ensemble predictions for this location
                        pixel_preds = ensemble_preds[:, b, t, h, w]

                        # Create histogram
                        hist = torch.histc(
                            pixel_preds, bins=num_bins, min=global_min, max=global_max
                        )
                        hist = hist / E  # Normalize to get probability

                        # Calculate entropy (avoiding log(0))
                        valid_probs = hist[hist > 0]
                        pixel_entropy = -torch.sum(valid_probs * torch.log2(valid_probs))
                        entropy[b, t, h, w] = pixel_entropy

        return entropy


class UncertaintyEstimator:
    def __init__(self, config):
        """
        Initialize uncertainty estimator

        Args:
            config: Configuration with model parameters
        """
        self.config = config

    def estimate_uncertainty(self, ensemble_predictions):
        """
        Estimate uncertainty from ensemble predictions

        Args:
            ensemble_predictions: Dictionary of ensemble predictions,
                                 each with shape [E, B, T, H, W]

        Returns:
            Dictionary of uncertainty metrics for each field
        """
        uncertainties = {}

        for field, preds in ensemble_predictions.items():
            # Calculate mean and variance across ensemble dimension
            mean_pred = preds.mean(dim=0)  # [B, T, H, W]
            variance = ((preds - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)  # [B, T, H, W]

            # Calculate additional uncertainty metrics
            ensemble_spread = preds.std(dim=0)  # Standard deviation across ensemble
            entropy = self._calculate_entropy(preds)  # Information entropy

            uncertainties[field] = {
                "variance": variance,
                "spread": ensemble_spread,
                "entropy": entropy,
            }

        return uncertainties

    def _calculate_entropy(self, ensemble_preds, num_bins=10):
        """
        Calculate entropy by binning predictions and computing Shannon entropy

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            num_bins: Number of bins for discretization

        Returns:
            Entropy map with shape [B, T, H, W]
        """
        # For continuous variables, we discretize into bins
        E, B, T, H, W = ensemble_preds.shape

        # Reshape for binning
        flat_preds = ensemble_preds.reshape(E, -1)

        # Find global min and max for consistent binning
        global_min = flat_preds.min()
        global_max = flat_preds.max()
        bin_width = (global_max - global_min) / num_bins

        # Initialize entropy map
        entropy = torch.zeros(B, T, H, W)

        # Calculate entropy for each pixel location
        for b in range(B):
            for t in range(T):
                for h in range(H):
                    for w in range(W):
                        # Get ensemble predictions for this location
                        pixel_preds = ensemble_preds[:, b, t, h, w]

                        # Create histogram
                        hist = torch.histc(
                            pixel_preds, bins=num_bins, min=global_min, max=global_max
                        )
                        hist = hist / E  # Normalize to get probability

                        # Calculate entropy (avoiding log(0))
                        valid_probs = hist[hist > 0]
                        pixel_entropy = -torch.sum(valid_probs * torch.log2(valid_probs))
                        entropy[b, t, h, w] = pixel_entropy

        return entropy

    def optimized_entropy(self, ensemble_preds, num_bins=10):
        """
        Vectorized implementation of entropy calculation for better performance

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            num_bins: Number of bins for discretization

        Returns:
            Entropy map with shape [B, T, H, W]
        """
        E, B, T, H, W = ensemble_preds.shape

        # Find global min and max for consistent binning
        global_min = ensemble_preds.min()
        global_max = ensemble_preds.max()

        # Reshape to [E, B*T*H*W]
        reshaped_preds = ensemble_preds.reshape(E, -1)

        # Compute entropy for each spatial-temporal location
        entropy = torch.zeros(B * T * H * W)

        for i in range(B * T * H * W):
            # Get predictions for this location
            pixel_preds = reshaped_preds[:, i]

            # Create histogram
            hist = torch.histc(pixel_preds, bins=num_bins, min=global_min, max=global_max)
            hist = hist / E  # Normalize

            # Calculate entropy
            valid_probs = hist[hist > 0]
            pixel_entropy = -torch.sum(valid_probs * torch.log2(valid_probs))
            entropy[i] = pixel_entropy

        # Reshape back to [B, T, H, W]
        return entropy.reshape(B, T, H, W)


class CalibrationMetrics:
    """
    Calculate calibration metrics for probabilistic forecasts
    """

    def __init__(self):
        pass

    def rank_histogram(self, ensemble_preds, observations, bins=10):
        """
        Calculate rank histogram (verification rank histogram)

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            observations: Ground truth observations with shape [B, T, H, W]
            bins: Number of bins for the histogram

        Returns:
            Rank histogram
        """
        E, B, T, H, W = ensemble_preds.shape

        # Reshape for easier processing
        ensemble_flat = ensemble_preds.reshape(E, -1)
        obs_flat = observations.reshape(-1)

        # Count ranks
        ranks = torch.zeros(E + 1)

        for i in range(len(obs_flat)):
            rank = torch.sum(ensemble_flat[:, i] < obs_flat[i])
            ranks[rank] += 1

        # Normalize
        ranks = ranks / ranks.sum()

        return ranks

    def crps(self, ensemble_preds, observations):
        """
        Calculate Continuous Ranked Probability Score (CRPS)

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            observations: Ground truth observations with shape [B, T, H, W]

        Returns:
            CRPS score
        """
        E, B, T, H, W = ensemble_preds.shape

        # Reshape
        ensemble_flat = ensemble_preds.reshape(E, -1)
        obs_flat = observations.reshape(-1)

        # Find global min and max for consistent intervals
        global_min = ensemble_flat.min()
        global_max = ensemble_flat.max()

        crps_values = torch.zeros(len(obs_flat))

        for i in range(len(obs_flat)):
            # Sort ensemble members
            sorted_ensemble = torch.sort(ensemble_flat[:, i])[0]

            # Calculate CRPS
            obs = obs_flat[i]
            crps_sum = 0.0

            # Implementation of CRPS formula
            for j in range(E):
                # Heaviside step function
                heaviside = (sorted_ensemble[j] >= obs).float()

                # Calculate squared difference between CDF and observation
                if j == 0:
                    prev_ens = sorted_ensemble[0]
                    crps_sum += ((0.0 - heaviside) ** 2) * (sorted_ensemble[0] - global_min)
                else:
                    crps_sum += ((j / E - heaviside) ** 2) * (
                        sorted_ensemble[j] - sorted_ensemble[j - 1]
                    )

            # Add last interval
            crps_sum += ((1.0 - 1.0) ** 2) * (global_max - sorted_ensemble[-1])

            crps_values[i] = crps_sum

        # Return mean CRPS
        return crps_values.mean()

    def spread_skill_ratio(self, ensemble_preds, observations):
        """
        Calculate spread-skill ratio (ratio of ensemble spread to RMSE)

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            observations: Ground truth observations with shape [B, T, H, W]

        Returns:
            Spread-skill ratio
        """
        # Calculate ensemble mean
        ensemble_mean = ensemble_preds.mean(dim=0)  # [B, T, H, W]

        # Calculate ensemble spread (standard deviation)
        ensemble_spread = ensemble_preds.std(dim=0)  # [B, T, H, W]

        # Calculate RMSE between mean prediction and observations
        rmse = torch.sqrt(((ensemble_mean - observations) ** 2).mean())

        # Calculate mean spread
        mean_spread = ensemble_spread.mean()

        # Calculate ratio
        ratio = mean_spread / rmse

        return ratio

    def pit_histogram(self, ensemble_preds, observations, bins=10):
        """
        Calculate Probability Integral Transform (PIT) histogram

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            observations: Ground truth observations with shape [B, T, H, W]
            bins: Number of bins for the histogram

        Returns:
            PIT histogram
        """
        E, B, T, H, W = ensemble_preds.shape

        # Reshape
        ensemble_flat = ensemble_preds.reshape(E, -1)
        obs_flat = observations.reshape(-1)

        pit_values = torch.zeros(len(obs_flat))

        for i in range(len(obs_flat)):
            # Sort ensemble members
            sorted_ensemble = torch.sort(ensemble_flat[:, i])[0]

            # Calculate empirical CDF value at observation
            obs = obs_flat[i]
            cdf_value = torch.sum(sorted_ensemble <= obs).float() / E

            # Store PIT value
            pit_values[i] = cdf_value

        # Create histogram
        hist = torch.histc(pit_values, bins=bins, min=0, max=1)
        hist = hist / hist.sum()

        return hist


class EnsemblePostProcessor:
    """
    Post-process ensemble predictions to improve calibration
    """

    def __init__(self, config):
        """
        Initialize post-processor

        Args:
            config: Configuration with post-processing parameters
        """
        self.config = config

    def bias_correction(self, ensemble_preds, bias_field):
        """
        Apply bias correction to ensemble predictions

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            bias_field: Bias field with shape [B, T, H, W]

        Returns:
            Bias-corrected ensemble predictions
        """
        # Apply bias correction to each ensemble member
        corrected_preds = ensemble_preds - bias_field.unsqueeze(0)

        return corrected_preds

    def variance_inflation(self, ensemble_preds, inflation_factor):
        """
        Apply variance inflation to ensemble predictions

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            inflation_factor: Factor to inflate the variance

        Returns:
            Variance-inflated ensemble predictions
        """
        # Calculate ensemble mean
        ensemble_mean = ensemble_preds.mean(dim=0)  # [B, T, H, W]

        # Calculate deviations from mean
        deviations = ensemble_preds - ensemble_mean.unsqueeze(0)

        # Apply inflation
        inflated_preds = ensemble_mean.unsqueeze(0) + deviations * math.sqrt(inflation_factor)

        return inflated_preds

    def quantile_mapping(self, ensemble_preds, target_distribution, num_quantiles=100):
        """
        Apply quantile mapping to map ensemble distribution to a target distribution

        Args:
            ensemble_preds: Ensemble predictions with shape [E, B, T, H, W]
            target_distribution: Function that generates samples from target distribution
            num_quantiles: Number of quantiles for mapping

        Returns:
            Mapped ensemble predictions
        """
        E, B, T, H, W = ensemble_preds.shape

        # Initialize output
        mapped_preds = torch.zeros_like(ensemble_preds)

        # Process each location separately
        for b in range(B):
            for t in range(T):
                for h in range(H):
                    for w in range(W):
                        # Get ensemble predictions for this location
                        preds = ensemble_preds[:, b, t, h, w]

                        # Sort predictions
                        sorted_preds, indices = torch.sort(preds)

                        # Generate samples from target distribution
                        target_samples = target_distribution(E)
                        sorted_target, _ = torch.sort(target_samples)

                        # Map values
                        for e in range(E):
                            mapped_preds[indices[e], b, t, h, w] = sorted_target[e]

        return mapped_preds


def calculate_global_metrics(uncertainty_maps, observations):
    """
    Calculate global uncertainty metrics across the entire domain

    Args:
        uncertainty_maps: Dictionary of uncertainty metrics for each field
        observations: Ground truth observations

    Returns:
        Dictionary of global metrics
    """
    global_metrics = {}

    for field, metrics in uncertainty_maps.items():
        global_metrics[field] = {
            "mean_variance": metrics["variance"].mean().item(),
            "mean_spread": metrics["spread"].mean().item(),
            "mean_entropy": metrics["entropy"].mean().item(),
            "max_variance": metrics["variance"].max().item(),
            "min_variance": metrics["variance"].min().item(),
        }

    return global_metrics
