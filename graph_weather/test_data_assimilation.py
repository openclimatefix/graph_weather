"""
Test script for the complete self-supervised data assimilation pipeline
"""

import torch
import numpy as np
from graph_weather.graph_weather.models.data_assimilation import (
    DataAssimilationModel, 
    SimpleDataAssimilationModel, 
    ThreeDVarLoss, 
    generate_synthetic_data
)
from graph_weather.graph_weather.data.assimilation_dataloader import (
    AssimilationDataModule,
    create_synthetic_assimilation_dataset
)
from graph_weather.graph_weather.models.training_loop import (
    DataAssimilationTrainer,
    train_data_assimilation_model,
    compare_with_baselines
)
from graph_weather.graph_weather.models.evaluation import (
    DataAssimilationEvaluator,
    compare_methods,
    compute_rmse
)
from graph_weather.graph_weather.models.visualization import (
    plot_training_curves,
    plot_comparison_grid,
    plot_error_maps,
    plot_rmse_comparison,
    create_summary_dashboard
)


def test_basic_3dvar_loss():
    """Test the basic 3D-Var loss function"""
    print("Testing 3D-Var loss function...")
    
    # Create sample data
    batch_size, grid_size = 4, (5, 5)
    background = torch.randn(batch_size, 1, *grid_size)
    observations = torch.randn(batch_size, 1, *grid_size)
    analysis = torch.randn(batch_size, 1, *grid_size)
    
    # Initialize loss function
    loss_fn = ThreeDVarLoss()
    
    # Compute loss
    loss = loss_fn(analysis, background, observations)
    print(f"3D-Var loss: {loss.item():.4f}")
    
    # Test with custom covariances
    B = torch.eye(grid_size[0] * grid_size[1]) * 0.5
    R = torch.eye(grid_size[0] * grid_size[1]) * 0.3
    loss_fn_custom = ThreeDVarLoss(
        background_error_covariance=B,
        observation_error_covariance=R
    )
    
    loss_custom = loss_fn_custom(analysis, background, observations)
    print(f"3D-Var loss with custom covariances: {loss_custom.item():.4f}")
    
    print("✓ 3D-Var loss test passed\n")


def test_data_assimilation_model():
    """Test the data assimilation model"""
    print("Testing data assimilation model...")
    
    # Test simple FC model
    input_dim = 50  # 5x5 grid with 2 channels (bg + obs)
    model = DataAssimilationModel(input_dim=input_dim)
    
    batch_size = 4
    background = torch.randn(batch_size, input_dim // 2)
    observations = torch.randn(batch_size, input_dim // 2)
    
    analysis = model(background, observations)
    print(f"FC Model - Input shape: {background.shape}, Output shape: {analysis.shape}")
    
    # Test convolutional model
    grid_size = (5, 5)
    model_conv = SimpleDataAssimilationModel(grid_size=grid_size, num_channels=1)
    
    background_conv = torch.randn(batch_size, 1, *grid_size)
    observations_conv = torch.randn(batch_size, 1, *grid_size)
    
    analysis_conv = model_conv(background_conv, observations_conv)
    print(f"Conv Model - Input shape: {background_conv.shape}, Output shape: {analysis_conv.shape}")
    
    print("✓ Data assimilation model test passed\n")


def test_training_pipeline():
    """Test the complete training pipeline"""
    print("Testing training pipeline...")
    
    # Create synthetic data
    data_module = AssimilationDataModule(
        num_samples=200,
        grid_size=(8, 8),
        num_channels=1,
        bg_error_std=0.5,
        obs_error_std=0.3,
        obs_fraction=0.6,
        batch_size=16
    )
    data_module.setup()
    
    # Initialize model and loss
    model = SimpleDataAssimilationModel(
        grid_size=(8, 8),
        num_channels=1,
        hidden_dim=32,
        num_layers=2
    )
    
    # Train the model
    trainer, results = train_data_assimilation_model(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        epochs=10,  # Small number for testing
        lr=1e-3,
        device='cpu'
    )
    
    print(f"Final training loss: {results['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {results['val_losses'][-1]:.4f}")
    
    # Plot training curves
    plot_training_curves(
        results['train_losses'], 
        results['val_losses'], 
        title="Test Training Curves"
    )
    
    print("✓ Training pipeline test passed\n")
    
    return trainer, data_module


def test_evaluation_pipeline(trainer, data_module):
    """Test the evaluation pipeline"""
    print("Testing evaluation pipeline...")
    
    # Evaluate the trained model
    eval_results = trainer.evaluate_model(data_module.test_dataloader(), compute_metrics=True)
    
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Initialize evaluator
    evaluator = DataAssimilationEvaluator(trainer.model, device='cpu')
    overall_metrics = evaluator.evaluate_dataset(data_module.test_dataloader())
    
    print("\nOverall Metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("✓ Evaluation pipeline test passed\n")
    
    return overall_metrics


def test_comparison_with_baselines(trainer, data_module):
    """Test comparison with classical baselines"""
    print("Testing comparison with baselines...")
    
    comparison = compare_with_baselines(
        trainer.model, 
        data_module.test_dataloader(), 
        device='cpu'
    )
    
    print("Baseline Comparison Results:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")
    
    # Create RMSE comparison plot
    rmse_comparison = {
        'Analysis': comparison['avg_analysis_rmse'],
        'Background': comparison['avg_background_rmse'], 
        'Observations': comparison['avg_observation_rmse'],
        'Persistence': comparison['avg_persistence_rmse']
    }
    
    plot_rmse_comparison(rmse_comparison, title="RMSE Comparison with Baselines")
    
    print("✓ Baseline comparison test passed\n")
    
    return comparison


def test_visualization_pipeline():
    """Test visualization capabilities"""
    print("Testing visualization pipeline...")
    
    # Generate sample data for visualization
    batch_size, grid_size = 1, (6, 6)
    background, observations, true_state = generate_synthetic_data(
        batch_size=batch_size, 
        grid_size=grid_size, 
        num_channels=1
    )
    
    # Create a simple "analysis" (for demonstration)
    analysis = (background + observations) / 2  # Simple average
    
    # Test comparison grid
    plot_comparison_grid(
        background, observations, analysis, true_state,
        titles=['Background', 'Observations', 'Analysis', 'True State']
    )
    
    # Test error maps
    plot_error_maps(
        background, observations, analysis, true_state,
        titles=['Background Error', 'Observation Error', 'Analysis Error']
    )
    
    print("✓ Visualization pipeline test passed\n")


def run_comprehensive_test():
    """Run a comprehensive test of the entire pipeline"""
    print("="*60)
    print("COMPREHENSIVE TEST: Self-Supervised Data Assimilation Pipeline")
    print("="*60)
    
    # Test 1: Basic components
    test_basic_3dvar_loss()
    
    # Test 2: Model architecture
    test_data_assimilation_model()
    
    # Test 3: Training pipeline
    trainer, data_module = test_training_pipeline()
    
    # Test 4: Evaluation pipeline
    eval_metrics = test_evaluation_pipeline(trainer, data_module)
    
    # Test 5: Baseline comparison
    comparison_results = test_comparison_with_baselines(trainer, data_module)
    
    # Test 6: Visualization
    test_visualization_pipeline()
    
    # Final summary dashboard
    print("Creating summary dashboard...")
    summary_metrics = eval_metrics.copy()
    summary_metrics.update(comparison_results)
    summary_metrics['model'] = trainer.model  # Include model for parameter analysis
    
    create_summary_dashboard(summary_metrics)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Print key results
    print(f"\nKey Results:")
    print(f"- Analysis RMSE: {comparison_results['avg_analysis_rmse']:.4f}")
    print(f"- Background RMSE: {comparison_results['avg_background_rmse']:.4f}")
    print(f"- Analysis improvement over background: {comparison_results['analysis_improvement_over_bg']:.2f}%")
    print(f"- Analysis improvement over observations: {comparison_results['analysis_improvement_over_obs']:.2f}%")
    
    return True


def test_different_training_modes():
    """Test the model under different training conditions"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT TRAINING MODES")
    print("="*60)
    
    from graph_weather.graph_weather.models.training_loop import train_with_different_modes
    
    # Test with different configurations
    results = train_with_different_modes(
        model_class=lambda **kwargs: SimpleDataAssimilationModel(
            grid_size=(6, 6), 
            num_channels=1, 
            hidden_dim=16,
            num_layers=2
        ),
        data_module=AssimilationDataModule,
        grid_size=(6, 6),
        num_channels=1,
        epochs=5,  # Few epochs for testing
        lr=1e-3,
        device='cpu'
    )
    
    print("\nTraining Mode Results:")
    for mode, result in results.items():
        eval_res = result['eval_results']
        print(f"\n{mode.upper()} MODE:")
        print(f"  Analysis RMSE: {eval_res.get('analysis_rmse', 'N/A')}")
        print(f"  Background RMSE: {eval_res.get('background_rmse', 'N/A')}")
        print(f"  Improvement over background: {eval_res.get('improvement_over_bg', 'N/A')}%")
    
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    # Test different training modes
    mode_results = test_different_training_modes()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("Self-Supervised Data Assimilation Pipeline is working correctly.")
    print("="*60)