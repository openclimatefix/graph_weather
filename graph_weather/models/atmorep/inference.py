"""
This module contains functions for performing inference using a trained AtmoRep model.
Each function is designed to support the AtmoRep model and related configurations.
"""

import torch
from graph_weather.models.atmorep.config import AtmoRepConfig
from graph_weather.models.atmorep.model.atmorep import AtmoRep


def load_model(model_path, config=None, device="cuda", map_location=None):
    """
    Load a trained AtmoRep model

    Args:
        model_path: Path to the saved model checkpoint
        config: Optional configuration (if None, loaded from checkpoint)
        device: Device to load the model on
        map_location: Optional map_location argument for torch.load; if provided,
                      it will override the device parameter.

    Returns:
        Loaded model and configuration
    """
    if map_location is None:
        map_location = device
    else:
        device = map_location

    checkpoint = torch.load(model_path, map_location=map_location)
    if config is None:
        config = checkpoint.get("config")
    if config is None:
        raise ValueError("Config not found in checkpoint and not provided")

    # If config is a dictionary, instantiate the AtmoRepConfig class
    if isinstance(config, dict):
        config = AtmoRepConfig(**config)

    model = AtmoRep(config)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model, config


def inference(model, data, normalizer=None, ensemble_mode="mean"):
    """
    Perform inference with a trained AtmoRep model.

    Args:
        model: Trained MultiFormer model.
        data: Dictionary of field data, each with shape [B, T, H, W].
        normalizer: Optional FieldNormalizer for data normalization.
        ensemble_mode: How to combine ensemble predictions ('mean', 'median', or 'sample').

    Returns:
        Dictionary of predictions for each field with shape [B, T, H, W].
    """
    model.eval()

    # Normalize data if a normalizer is provided
    if normalizer:
        normalized_data = {
            field: normalizer.normalize(field_data, field) for field, field_data in data.items()
        }
    else:
        normalized_data = data

    # Determine device
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    patch_size = getattr(model.config, "patch_size", 16)
    # Create dummy masks with zeros by default (if your model requires them)
    masks = {
        field: torch.zeros(
            field_data.shape[0],
            field_data.shape[1],
            (field_data.shape[2] // patch_size) * (field_data.shape[3] // patch_size),
        )
        for field, field_data in normalized_data.items()
    }

    # Move data and masks to the same device as the model
    normalized_data = {k: v.to(device) for k, v in normalized_data.items()}
    masks = {k: v.to(device) for k, v in masks.items()}

    with torch.no_grad():
        # Direct model call
        predictions = model(normalized_data, masks)

    processed_predictions = {}
    for field, preds in predictions.items():
        # Handle ensemble dimensions if present
        if preds.dim() == 5:
            if ensemble_mode == "mean":
                field_pred = preds.mean(dim=0)
            elif ensemble_mode == "median":
                field_pred = preds.median(dim=0).values
            elif ensemble_mode == "sample":
                idx = torch.randint(0, preds.size(0), (1,)).item()
                field_pred = preds[idx]
            else:
                raise ValueError(f"Unknown ensemble mode: {ensemble_mode}")
        elif preds.dim() == 4:
            field_pred = preds
        else:
            raise ValueError("Unexpected prediction tensor dimensions")

        # Denormalize if a normalizer is provided
        if normalizer:
            field_pred = normalizer.denormalize(field_pred, field)

        processed_predictions[field] = field_pred

    return processed_predictions


def batch_inference(
    model, dataset, batch_size=16, normalizer=None, ensemble_mode="mean", num_workers=4
):
    """
    Perform batch inference on a dataset using the given model.

    Args:
        model: The trained model used for inference.
        dataset: The dataset to perform inference on.
        batch_size: The size of each batch for inference.
        normalizer: Optional normalizer to preprocess data.
        ensemble_mode: The method for combining predictions ('mean', 'median', or 'sample').
        num_workers: Number of workers to load the dataset in parallel (unused in this manual approach).

    Returns:
        A dictionary where keys are field names and values are tensors of predictions.
    """
    all_predictions = {}
    total = len(dataset)

    # Manual batching over dataset indices
    for i in range(0, total, batch_size):
        items = [dataset[j] for j in range(i, min(i + batch_size, total))]
        batch_data = {}

        # Collate items (assume each item is a dict of tensors with the same keys)
        for key in items[0]:
            stacked = torch.stack([item[key] for item in items])
            # If stacking results in extra dimensions, flatten them as needed
            if stacked.dim() == 5:
                B1, B2, T, H, W = stacked.shape
                stacked = stacked.view(B1 * B2, T, H, W)
            batch_data[key] = stacked

        # Run inference on the current batch
        preds = inference(model, batch_data, normalizer, ensemble_mode)
        for field, pred in preds.items():
            all_predictions.setdefault(field, []).append(pred)

    # Concatenate predictions across all batches
    for field in all_predictions:
        all_predictions[field] = torch.cat(all_predictions[field], dim=0)

    return all_predictions


def create_forecast(model, initial_data, steps=3):
    """
    Create a multi-step forecast using the model in an autoregressive fashion.

    Args:
        model: The AtmoRep model.
        initial_data: Dictionary with initial conditions {field: tensor(B, T, H, W)}.
        steps: Number of forecast steps beyond initial data.

    Returns:
        Dictionary with forecasted fields {field: tensor(B, T+steps, H, W)}.
    """
    model.eval()
    forecast_data = {k: v.clone() for k, v in initial_data.items()}

    # Ensure we have a time dimension
    for field in forecast_data:
        if forecast_data[field].dim() == 3:
            forecast_data[field] = forecast_data[field].unsqueeze(1)

    with torch.no_grad():
        for step in range(steps):
            next_step = model(forecast_data)
            for field in forecast_data:
                pred = next_step[field]
                # If there's an ensemble dimension, reduce it
                if pred.dim() == 5:
                    reduced = pred.mean(dim=0)
                    next_value = reduced[:, -1:, :, :]
                elif pred.dim() == 4:
                    next_value = pred[:, -1:, :, :]
                elif pred.dim() == 3:
                    next_value = pred.unsqueeze(1)
                else:
                    raise ValueError("Unexpected prediction dimensions in forecast")
                forecast_data[field] = torch.cat([forecast_data[field], next_value], dim=1)

    return forecast_data


def generate_training_masks(batch_data, config):
    """
    Generate random masks for training.

    Args:
        batch_data: Dictionary of input tensors.
        config: Configuration with mask parameters.

    Returns:
        Dictionary of mask tensors.
    """
    batch_masks = {}
    patch_size = getattr(config, "patch_size", 16)
    mask_ratio = getattr(config, "mask_ratio", 0.75)

    for field_name, field_data in batch_data.items():
        shape = tuple(field_data.shape)
        if len(shape) < 4:
            raise ValueError(f"Field '{field_name}' expected to be at least 4D, got shape {shape}")

        B, T, H, W = shape[:4]
        h_patches = H // patch_size
        w_patches = W // patch_size
        N = h_patches * w_patches

        mask = torch.zeros(B, T, N)
        for b in range(B):
            flat_mask = torch.zeros(T * N)
            num_tokens = flat_mask.numel()
            num_mask = int(num_tokens * mask_ratio)
            mask_indices = torch.randperm(num_tokens)[:num_mask]
            flat_mask[mask_indices] = 1
            mask[b] = flat_mask.view(T, N)

        batch_masks[field_name] = mask

    return batch_masks