"""
Visualization functions for self-supervised data assimilation
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

warnings.filterwarnings("ignore")


def plot_training_curves(train_losses, val_losses, title="Training Curves"):
    """
    Plot training and validation loss curves

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison_grid(
    background, observations, analysis, true_state=None, titles=None, figsize=(15, 10)
):
    """
    Plot a grid comparing background, observations, analysis, and true state

    Args:
        background: Background state
        observations: Observations
        analysis: Analysis from model
        true_state: True state (optional)
        titles: Titles for each subplot
        figsize: Figure size
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(background):
        background = background.cpu().numpy()
    if torch.is_tensor(observations):
        observations = observations.cpu().numpy()
    if torch.is_tensor(analysis):
        analysis = analysis.cpu().numpy()
    if true_state is not None and torch.is_tensor(true_state):
        true_state = true_state.cpu().numpy()

    if titles is None:
        titles = ["Background", "Observations", "Analysis"]
        if true_state is not None:
            titles.append("True State")

    n_plots = 3 if true_state is None else 4

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Determine common color scale
    all_data = [background, observations, analysis]
    if true_state is not None:
        all_data.append(true_state)

    # Handle different tensor shapes
    def get_data_for_plot(data):
        if data.ndim == 4:  # [batch, channels, height, width]
            return data[0, 0]  # Take first sample, first channel
        elif data.ndim == 3:  # [batch, height, width] or [channels, height, width]
            return data[0] if data.shape[0] <= 10 else data[0]  # Heuristic for batch vs channels
        elif data.ndim == 2:  # [height, width]
            return data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    processed_data = [get_data_for_plot(d) for d in all_data]
    vmin = min([d.min() for d in processed_data])
    vmax = max([d.max() for d in processed_data])

    # Plot each field
    im1 = axes[0].imshow(processed_data[0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0])
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(processed_data[1], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(processed_data[2], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[2].set_title(titles[2])
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    if true_state is not None and n_plots > 3:
        im4 = axes[3].imshow(processed_data[3], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[3].set_title(titles[3])
        axes[3].axis("off")
        plt.colorbar(im4, ax=axes[3])

    plt.tight_layout()
    plt.show()


def plot_error_maps(background, observations, analysis, true_state, titles=None, figsize=(18, 5)):
    """
    Plot error maps comparing different methods

    Args:
        background: Background state
        observations: Observations
        analysis: Analysis from model
        true_state: True state
        titles: Titles for each subplot
        figsize: Figure size
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(background):
        background = background.cpu().numpy()
    if torch.is_tensor(observations):
        observations = observations.cpu().numpy()
    if torch.is_tensor(analysis):
        analysis = analysis.cpu().numpy()
    if torch.is_tensor(true_state):
        true_state = true_state.cpu().numpy()

    if titles is None:
        titles = ["Background Error", "Observation Error", "Analysis Error"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Calculate errors
    def get_first_element(data):
        if data.ndim == 4:  # [batch, channels, height, width]
            return data[0, 0]  # Take first sample, first channel
        elif data.ndim == 3:  # [batch, height, width]
            return data[0]
        else:
            return data

    bg_error = get_first_element(background) - get_first_element(true_state)
    obs_error = get_first_element(observations) - get_first_element(true_state)
    analysis_error = get_first_element(analysis) - get_first_element(true_state)

    # Determine common color scale for errors (centered at 0)
    max_error = max(np.abs(bg_error).max(), np.abs(obs_error).max(), np.abs(analysis_error).max())

    # Plot error maps
    im1 = axes[0].imshow(
        bg_error if bg_error.ndim == 2 else bg_error[0],
        cmap="RdBu_r",
        vmin=-max_error,
        vmax=max_error,
    )
    axes[0].set_title(titles[0])
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        obs_error if obs_error.ndim == 2 else obs_error[0],
        cmap="RdBu_r",
        vmin=-max_error,
        vmax=max_error,
    )
    axes[1].set_title(titles[1])
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        analysis_error if analysis_error.ndim == 2 else analysis_error[0],
        cmap="RdBu_r",
        vmin=-max_error,
        vmax=max_error,
    )
    axes[2].set_title(titles[2])
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


def plot_rmse_comparison(metrics_dict, title="RMSE Comparison"):
    """
    Plot RMSE comparison between different methods

    Args:
        metrics_dict: Dictionary with method names as keys and RMSE values as values
        title: Title for the plot
    """
    methods = list(metrics_dict.keys())
    rmse_values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, rmse_values, color=["skyblue", "lightcoral", "lightgreen", "gold"])
    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_values) * 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def plot_improvement_heatmap(improvement_matrix, title="Improvement Heatmap"):
    """
    Plot improvement heatmap showing where analysis is better than background

    Args:
        improvement_matrix: Matrix showing improvement at each grid point
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        improvement_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Improvement"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_time_series_comparison(time_series_data, labels=None, title="Time Series Comparison"):
    """
    Plot time series comparison of metrics

    Args:
        time_series_data: List of time series to plot
        labels: Labels for each series
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))

    for i, series in enumerate(time_series_data):
        label = labels[i] if labels else f"Series {i+1}"
        plt.plot(series, label=label, linewidth=2)

    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_histogram_comparison(
    true_state, background, analysis, bins=50, title="Distribution Comparison"
):
    """
    Plot histogram comparison of distributions

    Args:
        true_state: True state values
        background: Background state values
        analysis: Analysis state values
        bins: Number of histogram bins
        title: Title for the plot
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(true_state):
        true_state = true_state.cpu().numpy()
    if torch.is_tensor(background):
        background = background.cpu().numpy()
    if torch.is_tensor(analysis):
        analysis = analysis.cpu().numpy()

    plt.figure(figsize=(10, 6))

    true_flat = true_state.flatten()
    bg_flat = background.flatten()
    analysis_flat = analysis.flatten()

    plt.hist(true_flat, bins=bins, alpha=0.5, label="True State", density=True)
    plt.hist(bg_flat, bins=bins, alpha=0.5, label="Background", density=True)
    plt.hist(analysis_flat, bins=bins, alpha=0.5, label="Analysis", density=True)

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scatter_comparison(true_state, background, analysis, title="Scatter Plot Comparison"):
    """
    Plot scatter comparison showing correlation between true and predicted values

    Args:
        true_state: True state values
        background: Background state values
        analysis: Analysis state values
        title: Title for the plot
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(true_state):
        true_state = true_state.cpu().numpy()
    if torch.is_tensor(background):
        background = background.cpu().numpy()
    if torch.is_tensor(analysis):
        analysis = analysis.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    true_flat = true_state.flatten()
    bg_flat = background.flatten()
    analysis_flat = analysis.flatten()

    # Background vs True
    axes[0].scatter(true_flat, bg_flat, alpha=0.5)
    min_val = min(true_flat.min(), bg_flat.min())
    max_val = max(true_flat.max(), bg_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    axes[0].set_xlabel("True State")
    axes[0].set_ylabel("Background")
    axes[0].set_title("Background vs True")
    axes[0].grid(True, alpha=0.3)

    # Analysis vs True
    axes[1].scatter(true_flat, analysis_flat, alpha=0.5)
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    axes[1].set_xlabel("True State")
    axes[1].set_ylabel("Analysis")
    axes[1].set_title("Analysis vs True")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_convergence_analysis(train_losses, val_losses, title="Convergence Analysis"):
    """
    Plot detailed convergence analysis

    Args:
        train_losses: Training losses
        val_losses: Validation losses
        title: Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(train_losses) + 1)

    # Training and validation loss
    axes[0, 0].plot(epochs, train_losses, label="Training Loss", color="blue")
    axes[0, 0].plot(epochs, val_losses, label="Validation Loss", color="red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Log scale
    axes[0, 1].semilogy(epochs, train_losses, label="Training Loss", color="blue")
    axes[0, 1].semilogy(epochs, val_losses, label="Validation Loss", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss (log scale)")
    axes[0, 1].set_title("Loss (Log Scale)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Loss difference
    loss_diff = np.array(train_losses) - np.array(val_losses)
    axes[1, 0].plot(epochs, loss_diff, color="purple")
    axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Training - Validation Loss")
    axes[1, 0].set_title("Overfitting Indicator")
    axes[1, 0].grid(True, alpha=0.3)

    # Improvement per epoch
    improvement = np.diff(train_losses)
    axes[1, 1].plot(epochs[1:], improvement, color="green")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss Improvement")
    axes[1, 1].set_title("Improvement per Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_parameter_analysis(model, title="Parameter Analysis"):
    """
    Plot analysis of model parameters

    Args:
        model: PyTorch model
        title: Title for the plot
    """
    param_norms = []
    param_names = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.data.norm().item()
            param_norms.append(param_norm)
            param_names.append(name)

    if not param_norms:  # Handle case where no parameters require gradients
        print("No parameters require gradients to visualize")
        return

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(param_names)), param_norms)
    plt.xlabel("Parameters")
    plt.ylabel("L2 Norm")
    plt.title(title)
    plt.xticks(
        range(len(param_names)),
        [name.split(".")[-1] for name in param_names],
        rotation=45,
        ha="right",
    )

    # Add value labels on bars
    for bar, value in zip(bars, param_norms):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(param_norms) * 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def create_summary_dashboard(metrics, figsize=(16, 12)):
    """
    Create a comprehensive dashboard summarizing all results

    Args:
        metrics: Dictionary with all evaluation metrics
        figsize: Figure size for the dashboard
    """
    fig = plt.figure(figsize=figsize)

    # Define grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Training curves (if available)
    if "train_losses" in metrics and "val_losses" in metrics:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics["train_losses"], label="Train", color="blue")
        ax1.plot(metrics["val_losses"], label="Val", color="red")
        ax1.set_title("Training Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. RMSE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_methods = []
    rmse_values = []
    for key, value in metrics.items():
        if "rmse" in key.lower():
            rmse_methods.append(key.replace("_rmse", "").replace("avg_", "").title())
            rmse_values.append(value)

    if rmse_methods:
        ax2.bar(rmse_methods, rmse_values, color=["skyblue", "lightcoral", "lightgreen"])
        ax2.set_title("RMSE Comparison")
        ax2.set_ylabel("RMSE")
        ax2.tick_params(axis="x", rotation=45)

    # 3. Correlation comparison
    ax3 = fig.add_subplot(gs[0, 2])
    corr_methods = []
    corr_values = []
    for key, value in metrics.items():
        if "correlation" in key.lower():
            corr_methods.append(key.replace("_correlation", "").replace("avg_", "").title())
            corr_values.append(value)

    if corr_methods:
        ax3.bar(corr_methods, corr_values, color=["gold", "orange"])
        ax3.set_title("Correlation Comparison")
        ax3.set_ylabel("Correlation")
        ax3.tick_params(axis="x", rotation=45)

    # 4. Bias comparison
    ax4 = fig.add_subplot(gs[1, 0])
    bias_methods = []
    bias_values = []
    for key, value in metrics.items():
        if "bias" in key.lower():
            bias_methods.append(key.replace("_bias", "").replace("avg_", "").title())
            bias_values.append(value)

    if bias_methods:
        ax4.bar(bias_methods, bias_values, color=["lightblue", "pink"])
        ax4.set_title("Bias Comparison")
        ax4.set_ylabel("Bias")
        ax4.tick_params(axis="x", rotation=45)
        ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # 5. Improvement metrics
    ax5 = fig.add_subplot(gs[1, 1])
    improvement_metrics = []
    improvement_values = []
    for key, value in metrics.items():
        if "improvement" in key.lower():
            improvement_metrics.append(key.replace("avg_", "").replace("_pct", "%").title())
            improvement_values.append(value)

    if improvement_metrics:
        bars = ax5.bar(
            improvement_metrics,
            improvement_values,
            color=["lightgreen" if v > 0 else "lightcoral" for v in improvement_values],
        )
        ax5.set_title("Improvement Metrics")
        ax5.set_ylabel("Improvement (%)")
        ax5.tick_params(axis="x", rotation=45)
        ax5.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # Add value labels
        for bar, value in zip(bars, improvement_values):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                height
                + (
                    max(improvement_values) * 0.01
                    if max(improvement_values) > 0
                    else min(improvement_values) * 0.01
                ),
                f"{value:.1f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
            )

    # 6. Information gain
    if "avg_information_gain" in metrics:
        ax6 = fig.add_subplot(gs[1, 2])
        info_gain = metrics["avg_information_gain"]
        ax6.bar(["Information Gain"], [info_gain], color="mediumpurple")
        ax6.set_title("Information Gain")
        ax6.set_ylabel("Gain (%)")
        ax6.text(
            0, info_gain + max(info_gain * 0.01, 0.1), f"{info_gain:.1f}%", ha="center", va="bottom"
        )

    # 7. Parameter norms (if model available)
    if "model" in metrics:
        ax7 = fig.add_subplot(gs[2, :])
        param_norms = []
        param_names = []
        for name, param in metrics["model"].named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm().item()
                param_norms.append(param_norm)
                param_names.append(name.split(".")[-1][:10])  # Shorten names

        if param_norms:  # Only plot if there are parameters to show
            # Only show top parameters to avoid overcrowding
            top_indices = np.argsort(param_norms)[-10:][::-1]  # Top 10 largest
            top_norms = [param_norms[i] for i in top_indices]
            top_names = [param_names[i] for i in top_indices]

            ax7.bar(top_names, top_norms, color="lightsteelblue")
            ax7.set_title("Top 10 Parameter Norms")
            ax7.set_ylabel("L2 Norm")
            ax7.tick_params(axis="x", rotation=45)
        else:
            ax7.text(
                0.5,
                0.5,
                "No parameters to display",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax7.transAxes,
                fontsize=14,
            )
            ax7.set_title("Parameter Norms")
            ax7.set_xticks([])
            ax7.set_yticks([])

    plt.suptitle("Data Assimilation Results Dashboard", fontsize=16)
    plt.show()


def visualize_observation_locations(observations, obs_mask, title="Observation Locations"):
    """
    Visualize where observations are available

    Args:
        observations: Observation tensor
        obs_mask: Boolean mask indicating observation locations
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))

    # Create a visualization where observed locations are highlighted
    obs_visual = (
        torch.zeros_like(observations[0, 0])
        if len(observations.shape) > 2
        else torch.zeros_like(observations[0])
    )
    obs_visual[obs_mask] = 1

    plt.imshow(obs_visual.cpu().numpy(), cmap="viridis", interpolation="none")
    plt.title(title)
    plt.colorbar(label="Observation Present (1) / Missing (0)")
    plt.show()
