import sys
import os
import torch
import unittest

# Add the AI assimilation module directory to the path
ai_assimilation_path = os.path.join(
    os.path.dirname(__file__), "..", "graph_weather", "models", "ai_assimilation"
)
sys.path.insert(0, ai_assimilation_path)

# Import the modules directly (avoiding package import issues)
import model
import loss
import data
import training


class TestAIAssimilation(unittest.TestCase):
    """Test class for AI-based data assimilation functionality."""

    def test_model_creation_and_forward_pass(self):
        """Test that the AI assimilation model can be created and performs forward pass."""
        state_size = 20
        net = model.AIAssimilationNet(state_size=state_size)

        # Create test inputs
        first_guess = torch.randn(3, state_size)
        observations = torch.randn(3, state_size)

        # Forward pass
        analysis = net(first_guess, observations)

        # Verify output shape and validity
        self.assertEqual(
            analysis.shape, (3, state_size), "Output shape should match input batch and state size"
        )
        self.assertFalse(torch.isnan(analysis).any().item(), "Output should not contain NaN values")
        self.assertFalse(torch.isinf(analysis).any().item(), "Output should not contain Inf values")

    def test_3dvar_loss_function(self):
        """Test that the 3D-Var loss function works correctly."""
        loss_fn = loss.ThreeDVarLoss()

        # Create test tensors
        batch_size = 2
        state_size = 15
        analysis = torch.randn(batch_size, state_size)
        first_guess = torch.randn(batch_size, state_size)
        observations = torch.randn(batch_size, state_size)

        # Calculate loss
        total_loss = loss_fn(analysis, first_guess, observations)

        # Verify loss properties
        self.assertEqual(total_loss.dim(), 0, "Loss should be a scalar tensor")
        self.assertGreaterEqual(total_loss.item(), 0, "Loss should be non-negative")
        self.assertFalse(torch.isnan(total_loss).any().item(), "Loss should not contain NaN values")
        self.assertFalse(torch.isinf(total_loss).any().item(), "Loss should not contain Inf values")

    def test_synthetic_data_generation(self):
        """Test that synthetic data generation works correctly."""
        # Generate synthetic data
        num_samples = 10
        state_size = 25
        first_guess, observations, true_state = data.generate_synthetic_assimilation_data(
            num_samples=num_samples,
            state_size=state_size,
            obs_fraction=0.6,
            bg_error_std=0.3,
            obs_error_std=0.2,
        )

        # Verify data shapes
        self.assertEqual(
            first_guess.shape, (num_samples, state_size), "First guess should have correct shape"
        )
        self.assertEqual(
            observations.shape, (num_samples, state_size), "Observations should have correct shape"
        )
        self.assertEqual(
            true_state.shape, (num_samples, state_size), "True state should have correct shape"
        )

        # Verify data validity
        self.assertFalse(
            torch.isnan(first_guess).any().item(), "First guess should not contain NaN values"
        )
        self.assertFalse(
            torch.isinf(first_guess).any().item(), "First guess should not contain Inf values"
        )
        self.assertFalse(
            torch.isnan(observations).any().item(), "Observations should not contain NaN values"
        )
        self.assertFalse(
            torch.isinf(observations).any().item(), "Observations should not contain Inf values"
        )

    def test_dataset_creation(self):
        """Test that the AI assimilation dataset works correctly."""
        # Generate test data
        num_samples = 8
        state_size = 12
        first_guess, observations, _ = data.generate_synthetic_assimilation_data(
            num_samples=num_samples, state_size=state_size
        )

        # Create dataset
        dataset = data.AIAssimilationDataset(first_guess, observations)

        # Verify dataset properties
        self.assertEqual(len(dataset), num_samples, "Dataset length should match number of samples")

        # Get a sample
        sample = dataset[0]

        # Verify sample structure
        self.assertIsInstance(sample, dict, "Sample should be a dictionary")
        self.assertIn("first_guess", sample, "Sample should contain 'first_guess'")
        self.assertIn("observations", sample, "Sample should contain 'observations'")

        # Verify sample shapes
        self.assertEqual(
            sample["first_guess"].shape,
            (state_size,),
            "First guess in sample should have correct shape",
        )
        self.assertEqual(
            sample["observations"].shape,
            (state_size,),
            "Observations in sample should have correct shape",
        )

    def test_trainer_functionality(self):
        """Test that the AI assimilation trainer works correctly."""
        state_size = 10

        # Create model and loss function
        net = model.AIAssimilationNet(state_size=state_size)
        loss_fn = loss.ThreeDVarLoss()

        # Create trainer
        trainer = training.AIBasedAssimilationTrainer(
            model=net, loss_fn=loss_fn, lr=1e-3, device="cpu"
        )

        # Create test batch
        batch_fg = torch.randn(2, state_size)
        batch_obs = torch.randn(2, state_size)

        # Run training step
        train_loss = trainer.train_step(batch_fg, batch_obs)

        # Verify training step result
        self.assertIsInstance(train_loss, float, "Training loss should be a float")
        self.assertFalse(
            torch.isnan(torch.tensor(train_loss)).any().item(), "Training loss should not be NaN"
        )
        self.assertFalse(
            torch.isinf(torch.tensor(train_loss)).any().item(), "Training loss should not be Inf"
        )


def run_tests():
    """Run all AI assimilation tests."""
    print("Running AI Assimilation Tests...\n")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIAssimilation)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("ALL TESTS PASSED! AI assimilation module is working correctly.")
    else:
        print(" SOME TESTS FAILED! Please check the AI assimilation module.")
        print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAI assimilation module verification completed successfully!")
        print("\nComponents tested:")
        print("- AIAssimilationNet (model): Passed")
        print("- ThreeDVarLoss (loss function): Passed")
        print("- AIAssimilationDataset (data handling): Passed")
        print("- AIBasedAssimilationTrainer (training): Passed")
        print("- Synthetic data generation: Passed")
    else:
        print("\n AI assimilation module has issues that need to be addressed.")
