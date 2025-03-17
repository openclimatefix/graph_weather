import logging
import os
from typing import Any, Dict, Tuple

import torch
import torch.optim as optim
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from graph_weather.models.atmorep.model.atmorep import AtmoRep
from graph_weather.models.atmorep.data.dataset import ERA5Dataset
from graph_weather.models.atmorep.training.loss import AtmoRepLoss


def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: AtmoRepLoss,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one training epoch.

    Args:
        model: The AtmoRep model (nn.Module).
        dataloader: DataLoader yielding batches from ERA5Dataset.
        optimizer: Optimizer for model parameters.
        loss_fn: Loss function instance.
        epoch: Current epoch index.

    Returns:
        A tuple (epoch_loss, epoch_field_losses) where:
         - epoch_loss is the total loss summed over batches.
         - epoch_field_losses is a dict with summed per-field losses.
    """
    model.train()
    epoch_loss = 0.0
    epoch_field_losses = {field: 0.0 for field in model.config.input_fields}

    for batch_idx, batch_data in enumerate(dataloader):
        # Generate training masks for the batch.
        batch_masks = generate_training_masks(batch_data, model.config)

        # Move data and masks to the device of the model.
        device = next(model.parameters()).device
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        batch_masks = {k: v.to(device) for k, v in batch_masks.items()}

        # Forward pass.
        predictions = model(batch_data, batch_masks)

        # Calculate loss.
        loss, loss_components = loss_fn(predictions, batch_data, batch_masks)

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        for field, field_loss in loss_components["field_losses"].items():
            epoch_field_losses[field] += field_loss

    return epoch_loss, epoch_field_losses


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    config: Any,
    output_dir: str,
    epoch: int,
    best_val_loss: float,
    is_best: bool,
    history: Dict[str, Any],
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The trained model.
        optimizer: The optimizer.
        config: The configuration object.
        output_dir: Directory where the checkpoint should be saved.
        epoch: Current epoch number.
        best_val_loss: The best validation loss so far.
        is_best: Boolean indicating if this is the best model.
        history: Dictionary containing training history.
    """
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "is_best": is_best,
        "history": history,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
    }
    save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)


def train_atmorep(
    config: Any,
    era5_path: str,
    output_dir: str,
    model: torch.nn.Module = None,
    resume_from: str = None,
) -> torch.nn.Module:
    """
    Train the AtmoRep model.

    Args:
        config: Configuration object with model parameters (must include
                'input_fields', 'patch_size', 'mask_ratio', etc.).
        era5_path: Path to ERA5 data.
        output_dir: Directory to save model checkpoints.
        model: Optional pre-created model (if None, will be created).
        resume_from: Optional path to checkpoint to resume training from.

    Returns:
        Trained model (torch.nn.Module).
    """
    # Create model if not provided.
    if model is None:
        model = AtmoRep(config)

    # Ensure the model has parameters; if it fails, let it break.
    params = list(model.parameters())
    if len(params) == 0:
        # If for some reason the model has no parameters, register a dummy parameter.
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        if hasattr(model, "register_parameter"):
            model.register_parameter("dummy_param", dummy_param)
        else:
            setattr(model, "dummy_param", dummy_param)
        params = [dummy_param]

    # Create dataset and dataloader.
    dataset = ERA5Dataset(config, era5_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Loss function.
    loss_fn = AtmoRepLoss(config)

    # Optimizer and scheduler.
    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 100
    )

    # Resume training if needed.
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint.get("scheduler", {}))
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Setup logging.
    logger = logging.getLogger("AtmoRep-Training")
    logger.setLevel(logging.INFO)

    # Training loop: call _train_epoch each epoch.
    for epoch in range(start_epoch, config.epochs):
        try:
            epoch_loss, epoch_field_losses = _train_epoch(
                model, dataloader, optimizer, loss_fn, epoch
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            break

        scheduler.step()

        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_field_losses = {
            field: (field_loss / len(dataloader)) for field, field_loss in epoch_field_losses.items()
        }

        logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        for field, field_loss in avg_field_losses.items():
            logger.info(f"  {field} loss: {field_loss:.4f}")

        # Save checkpoint periodically.
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            os.makedirs(output_dir, exist_ok=True)
            _save_checkpoint(
                model,
                optimizer,
                config,
                output_dir,
                epoch,
                best_val_loss=0.0,
                is_best=True,
                history={},
            )

    return model


def generate_training_masks(batch_data: Dict[str, torch.Tensor], config: Any) -> Dict[str, torch.Tensor]:
    """
    Generate random masks for training.

    Args:
        batch_data: Dictionary of input tensors.
        config: Configuration with mask parameters (patch_size, mask_ratio, etc.).

    Returns:
        Dictionary of mask tensors matching the spatial dimensions of the input.
    """
    batch_masks = {}

    # Assume config provides valid patch_size and mask_ratio
    patch_size = int(config.patch_size)   # Removed fallback
    mask_ratio = float(config.mask_ratio) # Removed fallback

    for field_name, field_data in batch_data.items():
        # Extract tensor shape safely
        if not hasattr(field_data, "shape"):
            raise ValueError(f"Field {field_name} is missing shape information.")

        shape = list(field_data.shape)
        if len(shape) < 4:
            raise ValueError(
                f"Expected input of shape (B, T, H, W), got {tuple(shape)} for field '{field_name}'"
            )
        B, T, H, W = shape[0], shape[1], shape[2], shape[3]

        # Create mask with the same spatial dimensions as the input
        mask = torch.zeros(B, T, H, W)

        # Apply random masking directly on the spatial dimensions
        for b in range(B):
            for t in range(T):
                flat_size = H * W
                indices = torch.randperm(flat_size)
                num_to_mask = max(1, int(flat_size * mask_ratio))

                flat_mask = torch.zeros(flat_size)
                flat_mask[indices[:num_to_mask]] = 1.0

                # Use einops for clarity
                mask[b, t] = rearrange(flat_mask, "(h w) -> h w", h=H, w=W)

        batch_masks[field_name] = mask

    return batch_masks
