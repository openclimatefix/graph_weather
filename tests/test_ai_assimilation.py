import torch
import pytest

from graph_weather.models.ai_assimilation import model, loss, data, training


def test_model_creation_and_forward_pass():
    state_size = 20
    net = model.AIAssimilationNet(state_size=state_size)

    # Create test inputs
    first_guess = torch.randn(3, state_size)
    observations = torch.randn(3, state_size)

    # Forward pass
    analysis = net(first_guess, observations)

    # Verify output shape and validity
    assert analysis.shape == (3, state_size), "Output shape should match input batch and state size"
    assert not torch.isnan(analysis).any().item(), "Output should not contain NaN values"
    assert not torch.isinf(analysis).any().item(), "Output should not contain Inf values"


def test_3dvar_loss_function():
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
    assert total_loss.dim() == 0, "Loss should be a scalar tensor"
    assert total_loss >= 0, "Loss should be non-negative"
    assert not torch.isnan(total_loss).any().item(), "Loss should not contain NaN values"
    assert not torch.isinf(total_loss).any().item(), "Loss should not contain Inf values"


def test_dataset_creation():
    # Create test data directly
    batch_size = 8
    state_size = 12

    first_guess = torch.randn(batch_size, state_size)
    observations = torch.randn(batch_size, state_size)

    # Create dataset
    dataset = data.AIAssimilationDataset(first_guess, observations)

    # Verify dataset properties
    assert len(dataset) == batch_size, "Dataset length should match number of samples"

    # Get a sample
    sample = dataset[0]

    # Verify sample structure
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "first_guess" in sample, "Sample should contain 'first_guess'"
    assert "observations" in sample, "Sample should contain 'observations'"

    # Verify sample shapes
    assert sample["first_guess"].shape == (
        state_size,
    ), "First guess in sample should have correct shape"
    assert sample["observations"].shape == (
        state_size,
    ), "Observations in sample should have correct shape"


def test_trainer_functionality():
    state_size = 10

    # Create model and loss function
    net = model.AIAssimilationNet(state_size=state_size)
    loss_fn = loss.ThreeDVarLoss()

    # Create trainer
    trainer = training.AIBasedAssimilationTrainer(model=net, loss_fn=loss_fn, lr=1e-3, device="cpu")

    # Create test batch
    batch_fg = torch.randn(2, state_size)
    batch_obs = torch.randn(2, state_size)

    # Run training step
    train_loss = trainer.train_step(batch_fg, batch_obs)

    # Verify training step result
    assert isinstance(train_loss, float), "Training loss should be a float"
    assert not torch.isnan(torch.tensor(train_loss)).any().item(), "Training loss should not be NaN"
    assert not torch.isinf(torch.tensor(train_loss)).any().item(), "Training loss should not be Inf"
