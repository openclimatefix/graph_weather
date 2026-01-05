"""
Evaluation metrics for self-supervised data assimilation
"""

import numpy as np
import torch


def compute_rmse(predictions, targets):
    """
    Compute Root Mean Square Error

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        rmse: Root mean square error
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        mae: Mean absolute error
    """
    return torch.mean(torch.abs(predictions - targets)).item()


def compute_bias(predictions, targets):
    """
    Compute bias (mean error)

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        bias: Mean error
    """
    return torch.mean(predictions - targets).item()


def compute_correlation(predictions, targets):
    """
    Compute Pearson correlation coefficient

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        correlation: Pearson correlation coefficient
    """
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)

    # Center the data
    pred_centered = pred_flat - torch.mean(pred_flat)
    target_centered = target_flat - torch.mean(target_flat)

    # Compute correlation
    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def compute_spatial_metrics(predictions, targets):
    """
    Compute spatial metrics for gridded data

    Args:
        predictions: Predicted values [batch, channels, height, width]
        targets: Target values [batch, channels, height, width]

    Returns:
        metrics: Dictionary with spatial metrics
    """
    batch_size, channels = predictions.shape[0], predictions.shape[1]

    rmse_spatial = []
    correlation_spatial = []

    for b in range(batch_size):
        for c in range(channels):
            pred_channel = predictions[b, c].flatten()
            target_channel = targets[b, c].flatten()

            rmse = torch.sqrt(torch.mean((pred_channel - target_channel) ** 2)).item()
            rmse_spatial.append(rmse)

            # Compute correlation
            pred_centered = pred_channel - torch.mean(pred_channel)
            target_centered = target_channel - torch.mean(target_channel)
            numerator = torch.sum(pred_centered * target_centered)
            denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2))

            if denominator != 0:
                corr = (numerator / denominator).item()
            else:
                corr = 0.0
            correlation_spatial.append(corr)

    return {
        "avg_rmse_spatial": np.mean(rmse_spatial),
        "std_rmse_spatial": np.std(rmse_spatial),
        "avg_correlation_spatial": np.mean(correlation_spatial),
        "std_correlation_spatial": np.std(correlation_spatial),
    }


def compute_information_gain(analysis, background, true_state):
    """
    Compute information gain from data assimilation
    Measures how much better the analysis is compared to background

    Args:
        analysis: Analysis state from model
        background: Background state (first guess)
        true_state: True state (for evaluation)

    Returns:
        info_gain: Information gain metric
    """
    bg_error = torch.mean((background - true_state) ** 2)
    analysis_error = torch.mean((analysis - true_state) ** 2)

    # Information gain as reduction in error variance
    info_gain = (bg_error - analysis_error) / bg_error * 100 if bg_error > 0 else 0

    return info_gain.item()


class DataAssimilationEvaluator:
    """
    Comprehensive evaluator for data assimilation models
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def evaluate_batch(self, batch):
        """
        Evaluate a single batch

        Args:
            batch: Dictionary with 'background', 'observations', 'true_state'

        Returns:
            metrics: Dictionary with evaluation metrics for the batch
        """
        self.model.eval()

        with torch.no_grad():
            background = batch["background"].to(self.device)
            observations = batch["observations"].to(self.device)
            true_state = batch["true_state"].to(self.device)

            # Get model analysis
            analysis = self.model(background, observations)

            # Compute metrics
            metrics = {
                "analysis_rmse": compute_rmse(analysis, true_state),
                "background_rmse": compute_rmse(background, true_state),
                "observations_rmse": compute_rmse(observations, true_state),
                "analysis_mae": compute_mae(analysis, true_state),
                "background_mae": compute_mae(background, true_state),
                "analysis_bias": compute_bias(analysis, true_state),
                "background_bias": compute_bias(background, true_state),
                "analysis_correlation": compute_correlation(analysis, true_state),
                "background_correlation": compute_correlation(background, true_state),
                "information_gain": compute_information_gain(analysis, background, true_state),
            }

            # Add spatial metrics if data is gridded
            if len(analysis.shape) > 2:  # Has spatial dimensions
                spatial_metrics = compute_spatial_metrics(analysis, true_state)
                metrics.update(spatial_metrics)

            return metrics

    def evaluate_dataset(self, data_loader):
        """
        Evaluate the model on an entire dataset

        Args:
            data_loader: DataLoader with test data

        Returns:
            overall_metrics: Dictionary with overall evaluation metrics
        """
        all_metrics = {
            "analysis_rmse": [],
            "background_rmse": [],
            "observations_rmse": [],
            "analysis_mae": [],
            "background_mae": [],
            "analysis_bias": [],
            "background_bias": [],
            "analysis_correlation": [],
            "background_correlation": [],
            "information_gain": [],
        }

        spatial_metrics_list = []

        for batch in data_loader:
            batch_metrics = self.evaluate_batch(batch)

            # Collect metrics
            for key in all_metrics.keys():
                if key in batch_metrics:
                    all_metrics[key].append(batch_metrics[key])

            # Collect spatial metrics if available
            if "avg_rmse_spatial" in batch_metrics:
                spatial_metrics_list.append(
                    {
                        "avg_rmse_spatial": batch_metrics["avg_rmse_spatial"],
                        "avg_correlation_spatial": batch_metrics["avg_correlation_spatial"],
                    }
                )

        # Compute overall metrics
        overall_metrics = {}
        for key, values in all_metrics.items():
            if values:  # Only compute if we have values
                overall_metrics[f"avg_{key}"] = np.mean(values)
                overall_metrics[f"std_{key}"] = np.std(values)

        # Compute spatial metrics
        if spatial_metrics_list:
            spatial_rmse_values = [m["avg_rmse_spatial"] for m in spatial_metrics_list]
            spatial_corr_values = [m["avg_correlation_spatial"] for m in spatial_metrics_list]

            overall_metrics["avg_spatial_rmse"] = np.mean(spatial_rmse_values)
            overall_metrics["std_spatial_rmse"] = np.std(spatial_rmse_values)
            overall_metrics["avg_spatial_correlation"] = np.mean(spatial_corr_values)
            overall_metrics["std_spatial_correlation"] = np.std(spatial_corr_values)

        return overall_metrics


def compare_methods(model_analysis, background, observations, true_state):
    """
    Compare different methods: model analysis, background, observations

    Args:
        model_analysis: Analysis from the trained model
        background: Background state
        observations: Observations
        true_state: True state for comparison

    Returns:
        comparison: Dictionary with comparison results
    """
    results = {}

    # Compute metrics for each method
    methods = {"analysis": model_analysis, "background": background, "observations": observations}

    for method_name, method_output in methods.items():
        results[f"{method_name}_rmse"] = compute_rmse(method_output, true_state)
        results[f"{method_name}_mae"] = compute_mae(method_output, true_state)
        results[f"{method_name}_bias"] = compute_bias(method_output, true_state)
        results[f"{method_name}_correlation"] = compute_correlation(method_output, true_state)

    # Compute improvements
    bg_rmse = results["background_rmse"]
    obs_rmse = results["observations_rmse"]
    analysis_rmse = results["analysis_rmse"]

    results["analysis_improvement_over_bg_pct"] = (
        ((bg_rmse - analysis_rmse) / bg_rmse * 100) if bg_rmse > 0 else 0
    )

    results["analysis_improvement_over_obs_pct"] = (
        ((obs_rmse - analysis_rmse) / obs_rmse * 100) if obs_rmse > 0 else 0
    )

    results["bg_improvement_over_obs_pct"] = (
        ((obs_rmse - bg_rmse) / obs_rmse * 100) if obs_rmse > 0 else 0
    )

    return results


def classical_3dvar_analysis(background, observations, H, B, R):
    """
    Classical 3D-Var analysis for comparison

    Args:
        background: Background state
        observations: Observations
        H: Observation operator
        B: Background error covariance
        R: Observation error covariance

    Returns:
        analysis: Classical 3D-Var analysis
    """
    # Reshape for matrix operations
    batch_size = background.shape[0]
    state_size = background[0].numel()
    obs_size = observations[0].numel()

    # Convert to appropriate shapes
    xb = background.view(batch_size, -1)  # [batch, state_size]
    y = observations.view(batch_size, -1)  # [batch, obs_size]

    analysis_results = []

    for i in range(batch_size):
        xb_i = xb[i : i + 1].T  # [state_size, 1]
        y_i = y[i : i + 1].T  # [obs_size, 1]

        # Compute Kalman gain: K = B * H^T * (H * B * H^T + R)^(-1)
        # For simplicity, using diagonal approximations
        if B is None:
            B_i = torch.eye(state_size, device=xb.device)
        else:
            B_i = B

        if R is None:
            R_i = torch.eye(obs_size, device=y.device)
        else:
            R_i = R

        if H is None:
            H_i = torch.eye(min(state_size, obs_size), device=xb.device)[:obs_size, :state_size]
        else:
            H_i = H

        # Calculate terms
        HBHT_R = torch.matmul(torch.matmul(H_i, B_i), H_i.T) + R_i
        K = torch.matmul(torch.matmul(B_i, H_i.T), torch.inverse(HBHT_R))

        # Compute analysis: xa = xb + K * (y - H * xb)
        innovation = y_i - torch.matmul(H_i, xb_i)
        correction = torch.matmul(K, innovation)
        xa_i = xb_i + correction

        analysis_results.append(xa_i.T)

    analysis = torch.cat(analysis_results, dim=0)
    return analysis.view_as(background)


def compute_cross_validation_score(model, data_loader, k_folds=5):
    """
    Compute cross-validation score for the model

    Args:
        model: Data assimilation model
        data_loader: Data loader
        k_folds: Number of folds for cross-validation

    Returns:
        cv_scores: List of scores for each fold
    """
    # For simplicity, using a basic approach to simulate cross-validation
    # In practice, you'd split your dataset into k folds
    model.eval()

    all_rmse = []
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            background = batch["background"].to(model.device if hasattr(model, "device") else "cpu")
            observations = batch["observations"].to(
                model.device if hasattr(model, "device") else "cpu"
            )

            if "true_state" in batch:
                true_state = batch["true_state"].to(
                    model.device if hasattr(model, "device") else "cpu"
                )

                analysis = model(background, observations)
                rmse = compute_rmse(analysis, true_state)
                all_rmse.append(rmse)
                batch_count += 1

                # Limit for efficiency
                if batch_count >= k_folds:
                    break

    return all_rmse


def compute_gradient_norm(model):
    """
    Compute the norm of gradients for the model

    Args:
        model: PyTorch model

    Returns:
        total_norm: Total gradient norm
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def compute_parameter_norm(model):
    """
    Compute the norm of parameters for the model

    Args:
        model: PyTorch model

    Returns:
        total_norm: Total parameter norm
    """
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm
