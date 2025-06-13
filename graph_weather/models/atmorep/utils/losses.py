import torch
from einops import rearrange


def rank_histogram(ensemble_preds, observations):
    """
    Calculate the verification rank histogram in a vectorized manner.

    Args:
        ensemble_preds (torch.Tensor): Ensemble predictions with shape [E, B, T, H, W].
        observations (torch.Tensor): Ground truth observations with shape [B, T, H, W].

    Returns:
        torch.Tensor: Normalized rank histogram of shape [E + 1].
    """
    E, B, T, H, W = ensemble_preds.shape
    # Rearrange tensors to flatten the spatial-temporal dimensions
    ensemble_flat = rearrange(ensemble_preds, "E B T H W -> E (B T H W)")
    obs_flat = rearrange(observations, "B T H W -> (B T H W)")

    # Count how many ensemble members are less than the observation for each location
    ranks = (ensemble_flat < obs_flat.unsqueeze(0)).sum(dim=0)
    histogram = torch.bincount(ranks, minlength=E + 1).float()
    histogram = histogram / histogram.sum()
    return histogram


def crps(ensemble_preds, observations):
    """
    Calculate the Continuous Ranked Probability Score (CRPS).

    Args:
        ensemble_preds (torch.Tensor): Ensemble predictions with shape [E, B, T, H, W].
        observations (torch.Tensor): Ground truth observations with shape [B, T, H, W].

    Returns:
        float: Mean CRPS score.
    """
    E, B, T, H, W = ensemble_preds.shape
    ensemble_flat = rearrange(ensemble_preds, "E B T H W -> E (B T H W)")
    obs_flat = rearrange(observations, "B T H W -> (B T H W)")

    global_min = ensemble_flat.min()
    global_max = ensemble_flat.max()

    crps_values = torch.zeros(len(obs_flat))
    # Loop over each pixel location; further vectorization could be explored
    for i in range(len(obs_flat)):
        sorted_ensemble = torch.sort(ensemble_flat[:, i])[0]
        obs = obs_flat[i]
        crps_sum = 0.0
        for j in range(E):
            heaviside = (sorted_ensemble[j] >= obs).float()
            if j == 0:
                crps_sum += ((0.0 - heaviside) ** 2) * (sorted_ensemble[0] - global_min)
            else:
                crps_sum += (((j / E) - heaviside) ** 2) * (
                    sorted_ensemble[j] - sorted_ensemble[j - 1]
                )
        # Last interval (contributes zero)
        crps_sum += ((1.0 - 1.0) ** 2) * (global_max - sorted_ensemble[-1])
        crps_values[i] = crps_sum

    return crps_values.mean()
