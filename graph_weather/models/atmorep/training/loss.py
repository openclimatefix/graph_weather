import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class MaskedReconstructionLoss(nn.Module):
    """
    Computes a masked MSE loss for each field and returns the average loss
    across all fields, as well as a dictionary of per-field losses.
    """
    def __init__(self, reduction: str = "none"):
        """
        Args:
            reduction (str): Reduction mode for MSELoss. Typically "none" if
                             you want to handle masking manually.
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        predictions: dict,
        targets: dict,
        masks: dict = None,
        field_weights: dict = None,
    ):
        """
        Args:
            predictions (dict): Dict of field predictions, each tensor is
                [E, B, T, H, W] or [B, T, H, W].
            targets (dict): Dict of field targets, each tensor is [B, T, H, W].
            masks (dict, optional): Dict of field masks, each tensor can be
                [B, T, N] or [B, T, H, W]. Defaults to None.
            field_weights (dict, optional): Dict mapping field names to scalar
                weights. If None, defaults to weight=1.0 for each field.

        Returns:
            avg_loss (torch.Tensor): Average loss across all fields.
            field_losses (dict): Dictionary of individual field losses.
        """
        total_loss = 0.0
        field_losses = {}
        num_fields = len(predictions)

        for field, field_preds in predictions.items():
            field_target = targets[field]
            field_mask = None
            if masks is not None:
                field_mask = masks.get(field, None)

            B, T, H, W = field_target.shape

            # -------------------------------------------------
            # Process the mask to get a spatial_mask of shape
            # [1, B, T, H, W], so it can be broadcast easily.
            # -------------------------------------------------
            if field_mask is not None:
                if field_mask.dim() == 4:
                    # Already a spatial mask: [B, T, H, W]
                    spatial_mask = rearrange(field_mask, "b t h w -> 1 b t h w")
                elif field_mask.dim() == 3:
                    # Patch masks: [B, T, N]. Assume N is a square number.
                    N = field_mask.shape[-1]
                    grid_size = int(math.sqrt(N))
                    spatial_mask = rearrange(
                        field_mask.float(),
                        "b t (x y) -> b t x y",
                        x=grid_size,
                        y=grid_size
                    )
                    spatial_mask = F.interpolate(
                        spatial_mask,
                        size=(H, W),
                        mode="nearest"
                    )
                    spatial_mask = rearrange(spatial_mask, "b t h w -> 1 b t h w")
                else:
                    raise ValueError("Unexpected mask dimensions.")
            else:
                # If no mask is provided, use a mask of all ones
                spatial_mask = torch.ones(
                    (1, B, T, H, W), device=field_target.device
                )

            # -------------------------------------------------
            # Expand the target if predictions have an
            # ensemble dimension [E, B, T, H, W].
            # -------------------------------------------------
            if field_preds.dim() == 5:
                # field_preds: [E, B, T, H, W]
                field_target_expanded = repeat(
                    field_target, "b t h w -> e b t h w", e=field_preds.shape[0]
                )
            elif field_preds.dim() == 4:
                # field_preds: [B, T, H, W]
                field_target_expanded = field_target
            else:
                raise ValueError("Unexpected prediction dimensions.")

            # -------------------------------------------------
            # Compute MSE and apply mask
            # -------------------------------------------------
            mse = self.mse_loss(field_preds, field_target_expanded)
            masked_mse = (mse * spatial_mask).sum() / (spatial_mask.sum() + 1e-8)

            # Apply any field-specific weighting
            weight = 1.0
            if field_weights is not None:
                weight = field_weights.get(field, 1.0)

            field_loss = weight * masked_mse
            field_losses[field] = field_loss
            total_loss += field_loss

        # Average over the number of predicted fields
        avg_loss = total_loss / num_fields

        return avg_loss, field_losses


class PhysicalConsistencyLoss(nn.Module):
    """
    Enforces simple physical constraints among fields, e.g.:
      - Non-negative wind speed from u10, v10
      - Penalizes large temperature gradients in t2m
    """
    def __init__(self, grad_threshold: float = 10.0):
        """
        Args:
            grad_threshold (float): Threshold for penalizing temperature
                                    gradients. Default is 10.0.
        """
        super().__init__()
        self.grad_threshold = grad_threshold

    def forward(self, predictions: dict):
        """
        Args:
            predictions (dict): Dict of field predictions, each with shape
                [E, B, T, H, W] or [B, T, H, W].
        Returns:
            loss (torch.Tensor): A scalar tensor with the physical constraint loss.
        """
        loss = 0.0

        # -------------------------------------------------
        # 1) Non-negative wind speed from u10 and v10
        # -------------------------------------------------
        if all(field in predictions for field in ["u10", "v10"]):
            u_wind = predictions["u10"].mean(dim=0)  # [B, T, H, W]
            v_wind = predictions["v10"].mean(dim=0)  # [B, T, H, W]
            wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
            # If wind_speed is negative, we want to penalize it (though it shouldn't be).
            negative_speed_loss = F.relu(-wind_speed).mean()
            loss += negative_speed_loss

        # -------------------------------------------------
        # 2) Penalize excessive temperature gradients
        # -------------------------------------------------
        if "t2m" in predictions:
            temp = predictions["t2m"].mean(dim=0)  # [B, T, H, W]

            # Horizontal gradients
            if temp.shape[-2] > 1 and temp.shape[-1] > 1:
                horiz_grad = (
                    torch.abs(temp[:, :, 1:, :] - temp[:, :, :-1, :]).mean()
                    + torch.abs(temp[:, :, :, 1:] - temp[:, :, :, :-1]).mean()
                )
            else:
                horiz_grad = torch.tensor(0.0, device=temp.device)

            # Temporal gradients
            if temp.shape[1] > 1:
                temp_grad = torch.abs(temp[:, 1:, :, :] - temp[:, :-1, :, :]).mean()
            else:
                temp_grad = torch.tensor(0.0, device=temp.device)

            # Penalize if gradients exceed a threshold
            excessive_grad_loss = (
                F.relu(horiz_grad - self.grad_threshold)
                + F.relu(temp_grad - self.grad_threshold)
            )
            loss += excessive_grad_loss

        return loss


class AtmoRepLoss(nn.Module):
    """
    Combines the reconstruction loss (with optional masks) and a physical
    consistency loss into one total loss.
    """
    def __init__(
        self,
        input_fields: list,
        recon_weight: float = 1.0,
        phys_weight: float = 0.1,
        field_weights: dict = None,
        grad_threshold: float = 10.0,
        mse_reduction: str = "none",
    ):
        """
        Args:
            input_fields (list): List of field names you expect as input.
            recon_weight (float): Weight for the reconstruction component.
            phys_weight (float): Weight for the physical consistency component.
            field_weights (dict): Optional dict mapping fields to scalar weights
                                  for the reconstruction loss.
            grad_threshold (float): Threshold for penalizing temperature gradients.
            mse_reduction (str): Reduction mode for the MSELoss used internally.
        """
        super().__init__()
        self.recon_loss = MaskedReconstructionLoss(reduction=mse_reduction)
        self.phys_loss = PhysicalConsistencyLoss(grad_threshold=grad_threshold)

        self.recon_weight = recon_weight
        self.phys_weight = phys_weight

        # Default to weight=1.0 for all fields if not provided
        self.field_weights = field_weights or {field: 1.0 for field in input_fields}

    def forward(
        self,
        predictions: dict,
        targets: dict,
        masks: dict = None
    ):
        """
        Args:
            predictions (dict): Dict of predicted fields, each with shape
                [E, B, T, H, W] or [B, T, H, W].
            targets (dict): Dict of target fields, each with shape [B, T, H, W].
            masks (dict, optional): Dict of masks for fields, each with shape
                [B, T, N] or [B, T, H, W]. Defaults to None.

        Returns:
            total_loss (torch.Tensor): Combined loss.
            loss_details (dict): Breakdown of loss components and field losses.
        """
        # Reconstruction loss
        recon_loss, field_losses = self.recon_loss(
            predictions,
            targets,
            masks,
            field_weights=self.field_weights
        )

        # Physical consistency loss
        phys_loss = self.phys_loss(predictions)

        # Weighted sum of both components
        total_loss = self.recon_weight * recon_loss + self.phys_weight * phys_loss

        return total_loss, {
            "reconstruction": recon_loss.item(),
            "physical": phys_loss.item(),
            "field_losses": {k: v.item() for k, v in field_losses.items()},
        }
