import math

import torch
from einops import rearrange


class UncertaintyEstimator:
    """
    Estimate uncertainty from ensemble predictions using vectorized operations.
    """

    def __init__(self, num_bins=10):
        """
        Initialize the UncertaintyEstimator.

        Args:
            num_bins (int): Number of bins for entropy calculation.
        """
        self.num_bins = num_bins

    def estimate_uncertainty(self, ensemble_predictions):
        """
        Estimate uncertainty metrics for each field.

        Args:
            ensemble_predictions (dict): Dictionary of ensemble predictions,
                                         each with shape [E, B, T, H, W].

        Returns:
            dict: Uncertainty metrics (variance, spread, and entropy) for each field.
        """
        uncertainties = {}

        for field, preds in ensemble_predictions.items():
            # Mean and variance across ensemble dimension
            mean_pred = preds.mean(dim=0)  # [B, T, H, W]
            variance = ((preds - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)
            ensemble_spread = preds.std(dim=0)
            entropy = self.optimized_entropy(preds, self.num_bins)

            uncertainties[field] = {
                "variance": variance,
                "spread": ensemble_spread,
                "entropy": entropy,
            }

        return uncertainties

    def optimized_entropy(self, ensemble_preds, num_bins):
        """
        Compute entropy in a vectorized fashion using einops for rearrangement.

        Args:
            ensemble_preds (torch.Tensor): Ensemble predictions with shape [E, B, T, H, W].
            num_bins (int): Number of bins for discretization.

        Returns:
            torch.Tensor: Entropy map with shape [B, T, H, W].
        """
        E, B, T, H, W = ensemble_preds.shape
        # Rearrange tensor to shape [E, (B*T*H*W)]
        preds_flat = rearrange(ensemble_preds, "E B T H W -> E (B T H W)")

        global_min = preds_flat.min()
        global_max = preds_flat.max()
        eps = 1e-8

        # Normalize predictions and compute bin indices
        norm = (preds_flat - global_min) / (global_max - global_min + eps)
        bin_indices = (norm * num_bins).floor().clamp(max=num_bins - 1).long()

        # One-hot encode and sum over ensemble dimension to get histogram counts
        one_hot = torch.nn.functional.one_hot(bin_indices, num_bins)
        hist = one_hot.sum(dim=0).float()  # shape: [(B*T*H*W), num_bins]
        prob = hist / E

        # Compute entropy (adding eps to avoid log2(0))
        entropy_flat = -(prob * torch.log2(prob + eps)).sum(dim=1)
        return entropy_flat.reshape(B, T, H, W)


class EnsemblePostProcessor:
    """
    Post-process ensemble predictions to improve calibration.
    """

    def __init__(self, bias_correction_value=0.0, inflation_factor=1.0):
        """
        Initialize the post-processor.

        Args:
            bias_correction_value (float): Value for bias correction.
            inflation_factor (float): Factor to inflate the ensemble variance.
        """
        self.bias_correction_value = bias_correction_value
        self.inflation_factor = inflation_factor

    def bias_correction(self, ensemble_preds, bias_field):
        """
        Apply bias correction to ensemble predictions.

        Args:
            ensemble_preds (torch.Tensor): Ensemble predictions with shape [E, B, T, H, W].
            bias_field (torch.Tensor): Bias field with shape [B, T, H, W].

        Returns:
            torch.Tensor: Bias-corrected ensemble predictions.
        """
        return ensemble_preds - bias_field.unsqueeze(0)

    def variance_inflation(self, ensemble_preds, inflation_factor=None):
        """
        Apply variance inflation to ensemble predictions.

        Args:
            ensemble_preds (torch.Tensor): Ensemble predictions with shape [E, B, T, H, W].
            inflation_factor (float, optional): Inflation factor to use. If None, uses instance value.

        Returns:
            torch.Tensor: Variance-inflated ensemble predictions.
        """
        if inflation_factor is None:
            inflation_factor = self.inflation_factor

        ensemble_mean = ensemble_preds.mean(dim=0)
        deviations = ensemble_preds - ensemble_mean.unsqueeze(0)
        return ensemble_mean.unsqueeze(0) + deviations * math.sqrt(inflation_factor)

    def quantile_mapping(self, ensemble_preds, target_distribution, num_quantiles=100):
        """
        Map the ensemble distribution to a target distribution using quantile mapping.

        Args:
            ensemble_preds (torch.Tensor): Ensemble predictions with shape [E, B, T, H, W].
            target_distribution (callable): Function that generates samples given a sample size.
            num_quantiles (int): Number of quantiles for mapping.

        Returns:
            torch.Tensor: Quantile-mapped ensemble predictions.
        """
        E, B, T, H, W = ensemble_preds.shape
        mapped_preds = torch.zeros_like(ensemble_preds)

        # Process each location separately; vectorization is challenging here
        for b in range(B):
            for t in range(T):
                for h in range(H):
                    for w in range(W):
                        preds = ensemble_preds[:, b, t, h, w]
                        sorted_preds, indices = torch.sort(preds)
                        target_samples = target_distribution(E)
                        sorted_target, _ = torch.sort(target_samples)
                        for e in range(E):
                            mapped_preds[indices[e], b, t, h, w] = sorted_target[e]
        return mapped_preds


def calculate_global_metrics(uncertainty_maps, observations):
    """
    Calculate global uncertainty metrics across all fields.

    Args:
        uncertainty_maps (dict): Dictionary of uncertainty metrics for each field.
        observations (torch.Tensor): Ground truth observations.

    Returns:
        dict: Global metrics for each field.
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
