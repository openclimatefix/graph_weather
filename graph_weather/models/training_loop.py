"""
Training loop for self-supervised data assimilation with 3D-Var loss
"""

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .data_assimilation import ThreeDVarLoss


class DataAssimilationTrainer:
    """Trainer for the self-supervised data assimilation model
    """

    def __init__(self, model, loss_fn, optimizer=None, lr=1e-3, device="cpu", scheduler=None):
        """Initialize the trainer.

        Args:
            model: Data assimilation model
            loss_fn: 3D-Var loss function
            optimizer: Optimizer (default: Adam)
            lr: Learning rate
            device: Device to train on
            scheduler: Learning rate scheduler
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device

        if optimizer is None:
            self.optimizer = Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []

    def train_step(self, background, observations):
        """
        Perform a single training step

        Args:
            background: Background state
            observations: Observations

        Returns:
            loss: Training loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        background = background.to(self.device)
        observations = observations.to(self.device)

        # Forward pass
        analysis = self.model(background, observations)

        # Compute loss
        loss = self.loss_fn(analysis, background, observations)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return loss.item()

    def validation_step(self, background, observations):
        """
        Perform a validation step

        Args:
            background: Background state
            observations: Observations

        Returns:
            loss: Validation loss value
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            background = background.to(self.device)
            observations = observations.to(self.device)

            # Forward pass
            analysis = self.model(background, observations)

            # Compute loss
            loss = self.loss_fn(analysis, background, observations)

        return loss.item()

    def train_epoch(self, train_loader):
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            avg_loss: Average training loss for the epoch
        """
        total_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            background = batch["background"]
            observations = batch["observations"]

            loss = self.train_step(background, observations)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, val_loader):
        """
        Validate for one epoch

        Args:
            val_loader: Validation data loader

        Returns:
            avg_loss: Average validation loss for the epoch
        """
        total_loss = 0
        num_batches = 0

        for batch in val_loader:
            background = batch["background"]
            observations = batch["observations"]

            loss = self.validation_step(background, observations)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=100,
        verbose=True,
        save_best_model=True,
        model_save_path="best_assimilation_model.pth",
    ):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            verbose: Whether to print progress
            save_best_model: Whether to save the best model
            model_save_path: Path to save the best model

        Returns:
            train_losses: Training losses for each epoch
            val_losses: Validation losses for each epoch
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if save_best_model and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        if save_best_model:
            self.model.load_state_dict(torch.load(model_save_path))

        return self.train_losses, self.val_losses

    def evaluate_model(self, test_loader, compute_metrics=True):
        """
        Evaluate the model on test data

        Args:
            test_loader: Test data loader
            compute_metrics: Whether to compute additional metrics

        Returns:
            results: Dictionary with evaluation results
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_analysis = []
        all_background = []
        all_observations = []
        all_true = []

        with torch.no_grad():
            for batch in test_loader:
                background = batch["background"].to(self.device)
                observations = batch["observations"].to(self.device)

                analysis = self.model(background, observations)
                loss = self.loss_fn(analysis, background, observations)

                total_loss += loss.item()
                num_batches += 1

                if compute_metrics:
                    all_analysis.append(analysis.cpu())
                    all_background.append(background.cpu())
                    all_observations.append(observations.cpu())

                    if "true_state" in batch:
                        all_true.append(batch["true_state"])

        avg_loss = total_loss / num_batches
        results = {"loss": avg_loss}

        if compute_metrics and all_true:
            # Compute additional metrics
            all_analysis = torch.cat(all_analysis, dim=0)
            all_background = torch.cat(all_background, dim=0)
            all_observations = torch.cat(all_observations, dim=0)
            all_true = torch.cat(all_true, dim=0)

            # RMSE metrics
            analysis_rmse = torch.sqrt(torch.mean((all_analysis - all_true) ** 2)).item()
            bg_rmse = torch.sqrt(torch.mean((all_background - all_true) ** 2)).item()
            obs_rmse = torch.sqrt(torch.mean((all_observations - all_true) ** 2)).item()

            results.update(
                {
                    "analysis_rmse": analysis_rmse,
                    "background_rmse": bg_rmse,
                    "observations_rmse": obs_rmse,
                    "improvement_over_bg": (
                        (bg_rmse - analysis_rmse) / bg_rmse * 100 if bg_rmse > 0 else 0
                    ),
                    "improvement_over_obs": (
                        (obs_rmse - analysis_rmse) / obs_rmse * 100 if obs_rmse > 0 else 0
                    ),
                }
            )

        return results


def train_data_assimilation_model(
    model,
    train_loader,
    val_loader,
    bg_error_covariance=None,
    obs_error_covariance=None,
    obs_operator=None,
    epochs=100,
    lr=1e-3,
    device="cpu",
):
    """
    Convenience function to train a data assimilation model

    Args:
        model: Data assimilation model
        train_loader: Training data loader
        val_loader: Validation data loader
        bg_error_covariance: Background error covariance
        obs_error_covariance: Observation error covariance
        obs_operator: Observation operator
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        trainer: Trained trainer object
        results: Training results
    """
    # Initialize the 3D-Var loss function
    loss_fn = ThreeDVarLoss(
        background_error_covariance=bg_error_covariance,
        observation_error_covariance=obs_error_covariance,
        observation_operator=obs_operator,
    )

    # Initialize the trainer
    trainer = DataAssimilationTrainer(model=model, loss_fn=loss_fn, lr=lr, device=device)

    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(trainer.optimizer, mode="min", factor=0.5, patience=10)
    trainer.scheduler = scheduler

    # Train the model
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=True
    )

    return trainer, {"train_losses": train_losses, "val_losses": val_losses}


def train_with_different_modes(
    model_class,
    data_module,
    input_dim=None,
    grid_size=None,
    num_channels=1,
    epochs=100,
    lr=1e-3,
    device="cpu",
):
    """Train the model in different modes:
    
    1. With good first guess (low background error)
    2. With poor first guess (high background error) - cold start
    3. With varying observation densities

    Args:
        model_class: Model class to instantiate
        data_module: Data module with different configurations
        input_dim: Input dimension for fully connected model
        grid_size: Grid size for convolutional model
        num_channels: Number of channels
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        results: Dictionary with results from different training modes
    """
    results = {}

    # Mode 1: With good first guess (low background error)
    print("Training with good first guess...")
    data_module_good_bg = data_module(
        num_samples=1000,
        grid_size=grid_size,
        num_channels=num_channels,
        bg_error_std=0.2,  # Low error
        obs_error_std=0.3,
        obs_fraction=0.5,
    )
    data_module_good_bg.setup()

    if input_dim:
        model_good_bg = model_class(input_dim=input_dim)
    else:
        model_good_bg = model_class(grid_size=grid_size, num_channels=num_channels)

    trainer_good_bg, res_good_bg = train_data_assimilation_model(
        model=model_good_bg,
        train_loader=data_module_good_bg.train_dataloader(),
        val_loader=data_module_good_bg.val_dataloader(),
        epochs=epochs,
        lr=lr,
        device=device,
    )

    results["good_bg"] = {
        "trainer": trainer_good_bg,
        "results": res_good_bg,
        "eval_results": trainer_good_bg.evaluate_model(data_module_good_bg.test_dataloader()),
    }

    # Mode 2: With poor first guess (high background error) - cold start
    print("Training with poor first guess (cold start)...")
    data_module_poor_bg = data_module(
        num_samples=1000,
        grid_size=grid_size,
        num_channels=num_channels,
        bg_error_std=1.0,  # High error
        obs_error_std=0.3,
        obs_fraction=0.5,
    )
    data_module_poor_bg.setup()

    if input_dim:
        model_poor_bg = model_class(input_dim=input_dim)
    else:
        model_poor_bg = model_class(grid_size=grid_size, num_channels=num_channels)

    trainer_poor_bg, res_poor_bg = train_data_assimilation_model(
        model=model_poor_bg,
        train_loader=data_module_poor_bg.train_dataloader(),
        val_loader=data_module_poor_bg.val_dataloader(),
        epochs=epochs,
        lr=lr,
        device=device,
    )

    results["poor_bg"] = {
        "trainer": trainer_poor_bg,
        "results": res_poor_bg,
        "eval_results": trainer_poor_bg.evaluate_model(data_module_poor_bg.test_dataloader()),
    }

    # Mode 3: With sparse observations
    print("Training with sparse observations...")
    data_module_sparse_obs = data_module(
        num_samples=1000,
        grid_size=grid_size,
        num_channels=num_channels,
        bg_error_std=0.5,
        obs_error_std=0.3,
        obs_fraction=0.2,  # Sparse observations
    )
    data_module_sparse_obs.setup()

    if input_dim:
        model_sparse_obs = model_class(input_dim=input_dim)
    else:
        model_sparse_obs = model_class(grid_size=grid_size, num_channels=num_channels)

    trainer_sparse_obs, res_sparse_obs = train_data_assimilation_model(
        model=model_sparse_obs,
        train_loader=data_module_sparse_obs.train_dataloader(),
        val_loader=data_module_sparse_obs.val_dataloader(),
        epochs=epochs,
        lr=lr,
        device=device,
    )

    results["sparse_obs"] = {
        "trainer": trainer_sparse_obs,
        "results": res_sparse_obs,
        "eval_results": trainer_sparse_obs.evaluate_model(data_module_sparse_obs.test_dataloader()),
    }

    return results


def compare_with_baselines(model, test_loader, device="cpu"):
    """
    Compare the trained model with classical baselines

    Args:
        model: Trained assimilation model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        comparison: Dictionary with comparison results
    """
    model.eval()
    results = {
        "analysis_rmse": [],
        "background_rmse": [],
        "observation_rmse": [],
        "persistence_rmse": [],
    }

    with torch.no_grad():
        for batch in test_loader:
            background = batch["background"].to(device)
            observations = batch["observations"].to(device)

            if "true_state" in batch:
                true_state = batch["true_state"].to(device)

                # Model analysis
                analysis = model(background, observations)

                # Compute RMSE for each method
                analysis_rmse = torch.sqrt(torch.mean((analysis - true_state) ** 2)).item()
                bg_rmse = torch.sqrt(torch.mean((background - true_state) ** 2)).item()
                obs_rmse = torch.sqrt(torch.mean((observations - true_state) ** 2)).item()

                # Persistence (assuming observations are closer to truth than background)
                # For simplicity, using a weighted average as persistence
                persistence = 0.7 * observations + 0.3 * background
                persist_rmse = torch.sqrt(torch.mean((persistence - true_state) ** 2)).item()

                results["analysis_rmse"].append(analysis_rmse)
                results["background_rmse"].append(bg_rmse)
                results["observation_rmse"].append(obs_rmse)
                results["persistence_rmse"].append(persist_rmse)

    # Compute averages
    comparison = {
        "avg_analysis_rmse": np.mean(results["analysis_rmse"]),
        "avg_background_rmse": np.mean(results["background_rmse"]),
        "avg_observation_rmse": np.mean(results["observation_rmse"]),
        "avg_persistence_rmse": np.mean(results["persistence_rmse"]),
        "analysis_improvement_over_bg": (
            (np.mean(results["background_rmse"]) - np.mean(results["analysis_rmse"]))
            / np.mean(results["background_rmse"])
            * 100
        ),
        "analysis_improvement_over_obs": (
            (np.mean(results["observation_rmse"]) - np.mean(results["analysis_rmse"]))
            / np.mean(results["observation_rmse"])
            * 100
        ),
    }

    return comparison
