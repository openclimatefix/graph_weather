"""
Example usage of the self-supervised data assimilation framework
"""

import numpy as np
import torch

from graph_weather.graph_weather.data.assimilation_dataloader import AssimilationDataModule
from graph_weather.graph_weather.models.data_assimilation import (
    SimpleDataAssimilationModel,
)
from graph_weather.graph_weather.models.evaluation import DataAssimilationEvaluator
from graph_weather.graph_weather.models.training_loop import train_data_assimilation_model
from graph_weather.graph_weather.models.visualization import (
    plot_comparison_grid,
    plot_error_maps,
    plot_training_curves,
)


def main():
    """Main function to demonstrate the self-supervised data assimilation framework."""
    print("Self-Supervised Data Assimilation Example")
    print("=" * 50)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define the problem setup
    print("\n1. Setting up the problem...")
    grid_size = (12, 12)  # 12x12 spatial grid
    num_channels = 1  # Single variable (e.g., temperature)
    batch_size = 16
    epochs = 20  # Small number for demo

    print(f"Grid size: {grid_size}")
    print(f"Number of channels: {num_channels}")
    print(f"Batch size: {batch_size}")
    print(f"Training epochs: {epochs}")

    # 2. Create data module
    print("\n2. Creating data module...")
    data_module = AssimilationDataModule(
        num_samples=500,  # Number of training samples
        grid_size=grid_size,
        num_channels=num_channels,
        bg_error_std=0.5,  # Background error standard deviation
        obs_error_std=0.3,  # Observation error standard deviation
        obs_fraction=0.6,  # 60% of grid points have observations
        batch_size=batch_size,
    )
    data_module.setup()

    print(f"Training samples: {len(data_module.train_dataloader().dataset)}")
    print(f"Validation samples: {len(data_module.val_dataloader().dataset)}")
    print(f"Test samples: {len(data_module.test_dataloader().dataset)}")

    # 3. Initialize the model
    print("\n3. Initializing the model...")
    model = SimpleDataAssimilationModel(
        grid_size=grid_size,
        num_channels=num_channels,
        hidden_dim=32,  # Hidden dimension for conv layers
        num_layers=2,  # Number of processing layers
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Train the model
    print("\n4. Training the model...")
    print("This will minimize the 3D-Var cost function without ground truth!")

    trainer, results = train_data_assimilation_model(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        epochs=epochs,
        lr=1e-3,
        device="cpu",  # Using CPU for this example
    )

    print(f"Final training loss: {results['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {results['val_losses'][-1]:.6f}")

    # 5. Plot training curves
    print("\n5. Plotting training curves...")
    plot_training_curves(
        results["train_losses"], results["val_losses"], title="Data Assimilation Training Curves"
    )

    # 6. Evaluate the model
    print("\n6. Evaluating the model...")
    evaluator = DataAssimilationEvaluator(model, device="cpu")
    eval_metrics = evaluator.evaluate_dataset(data_module.test_dataloader())

    print("Evaluation Metrics:")
    for key, value in eval_metrics.items():
        if "avg_" in key and (
            "rmse" in key or "mae" in key or "bias" in key or "correlation" in key
        ):
            print(f"  {key}: {value:.4f}")

    # 7. Visualize results
    print("\n7. Visualizing results...")

    # Get a batch from test data for visualization
    test_iter = iter(data_module.test_dataloader())
    batch = next(test_iter)

    background = batch["background"]
    observations = batch["observations"]

    # Generate analysis using the trained model
    model.eval()
    with torch.no_grad():
        analysis = model(background, observations)

    # If true state is available, visualize comparison
    if "true_state" in batch:
        true_state = batch["true_state"]

        print("Creating comparison visualization...")
        plot_comparison_grid(
            background,
            observations,
            analysis,
            true_state,
            titles=["Background", "Observations", "Analysis", "True State"],
        )

        print("Creating error maps...")
        plot_error_maps(
            background,
            observations,
            analysis,
            true_state,
            titles=["Background Error", "Observation Error", "Analysis Error"],
        )

    # 8. Compare with baselines
    print("\n8. Comparing with baselines...")
    from graph_weather.graph_weather.models.training_loop import compare_with_baselines

    comparison = compare_with_baselines(model, data_module.test_dataloader(), device="cpu")

    print("Baseline Comparison:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")

    # 9. Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("✓ Successfully trained a self-supervised data assimilation model")
    print("✓ Model learned to minimize 3D-Var cost function without ground truth")
    print(f"✓ Analysis RMSE: {comparison['avg_analysis_rmse']:.4f}")
    print(f"✓ Background RMSE: {comparison['avg_background_rmse']:.4f}")
    print(
        f"✓ Analysis improvement over background: {comparison['analysis_improvement_over_bg']:.2f}%"
    )
    improvement = comparison["analysis_improvement_over_obs"]
    print(f"✓ Analysis improvement over observations: {improvement:.2f}%")

    print("\nThe model successfully learned to combine background and observations")
    print("optimally to produce better analysis states than either input alone!")


if __name__ == "__main__":
    main()
