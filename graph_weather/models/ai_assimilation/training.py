from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .loss import ThreeDVarLoss


class AIBasedAssimilationTrainer(Module):

    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        device: str = "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()
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
        self.learning_rates = []

    def train_step(self, first_guess: torch.Tensor, observations: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        first_guess = first_guess.to(self.device)
        observations = observations.to(self.device)

        # Forward pass - get analysis from model
        analysis = self.model(first_guess, observations)

        # Compute 3D-Var loss
        loss = self.loss_fn(analysis, first_guess, observations)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return loss.item()

    def validation_step(self, first_guess: torch.Tensor, observations: torch.Tensor) -> float:
        """
        Perform a validation step.

        Args:
            first_guess: First-guess state
            observations: Observations

        Returns:
            loss: Validation loss value
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            first_guess = first_guess.to(self.device)
            observations = observations.to(self.device)

            # Forward pass
            analysis = self.model(first_guess, observations)

            # Compute loss
            loss = self.loss_fn(analysis, first_guess, observations)

        return loss.item()

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            first_guess = batch["first_guess"]
            observations = batch["observations"]

            loss = self.train_step(first_guess, observations)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            first_guess = batch["first_guess"]
            observations = batch["observations"]

            loss = self.validation_step(first_guess, observations)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        verbose: bool = True,
        save_best_model: bool = True,
        model_save_path: str = "best_ai_assimilation_model.pth",
        early_stopping_patience: int = 10,
    ) -> Tuple[list, list]:
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping and model saving
            if save_best_model and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": val_loss,
                    },
                    model_save_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}"
                )

        # Load best model if saved
        if save_best_model:
            checkpoint = torch.load(model_save_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return self.train_losses, self.val_losses

    def evaluate_model(
        self, test_loader: torch.utils.data.DataLoader, compute_additional_metrics: bool = True
    ) -> Dict[str, float]:

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_analysis = []
        all_first_guess = []
        all_observations = []

        with torch.no_grad():
            for batch in test_loader:
                first_guess = batch["first_guess"].to(self.device)
                observations = batch["observations"].to(self.device)

                analysis = self.model(first_guess, observations)
                loss = self.loss_fn(analysis, first_guess, observations)

                total_loss += loss.item()
                num_batches += 1

                if compute_additional_metrics:
                    all_analysis.append(analysis.cpu())
                    all_first_guess.append(first_guess.cpu())
                    all_observations.append(observations.cpu())

        avg_loss = total_loss / num_batches
        results = {"loss": avg_loss}

        if compute_additional_metrics and all_analysis:
            # Compute additional metrics by comparing with first-guess and observations
            all_analysis = torch.cat(all_analysis, dim=0)
            all_first_guess = torch.cat(all_first_guess, dim=0)
            all_observations = torch.cat(all_observations, dim=0)

            # Note: Since we're in self-supervised setting, we can't compute true RMSE
            # But we can compare the improvement in loss terms
            analysis_vs_bg_loss = self.loss_fn(
                all_analysis, all_first_guess, all_observations
            ).item()
            bg_vs_bg_loss = self.loss_fn(all_first_guess, all_first_guess, all_observations).item()

            results.update(
                {
                    "analysis_3dvar_loss": analysis_vs_bg_loss,
                    "first_guess_3dvar_loss": bg_vs_bg_loss,
                    "improvement_over_first_guess": (
                        (bg_vs_bg_loss - analysis_vs_bg_loss) / bg_vs_bg_loss * 100
                        if bg_vs_bg_loss > 0
                        else 0
                    ),
                }
            )

        return results


def train_ai_assimilation_model(
    model: Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    background_error_covariance: Optional[torch.Tensor] = None,
    observation_error_covariance: Optional[torch.Tensor] = None,
    observation_operator: Optional[torch.Tensor] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[Any, Dict[str, Any]]:
    # Initialize the 3D-Var loss function
    loss_fn = ThreeDVarLoss(
        background_error_covariance=background_error_covariance,
        observation_error_covariance=observation_error_covariance,
        observation_operator=observation_operator,
    )

    # Initialize the trainer
    trainer = AIBasedAssimilationTrainer(model=model, loss_fn=loss_fn, lr=lr, device=device)

    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(trainer.optimizer, mode="min", factor=0.5, patience=10)
    trainer.scheduler = scheduler

    # Train the model
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=True
    )

    return trainer, {"train_losses": train_losses, "val_losses": val_losses}


def plot_training_history(trainer: AIBasedAssimilationTrainer, title: str = "Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(trainer.train_losses) + 1)
    axes[0].plot(epochs, trainer.train_losses, label="Training Loss", color="blue")
    axes[0].plot(epochs, trainer.val_losses, label="Validation Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Losses")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot learning rates
    if trainer.learning_rates:
        axes[1].plot(epochs, trainer.learning_rates, label="Learning Rate", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title(f"{title} - Learning Rate")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
