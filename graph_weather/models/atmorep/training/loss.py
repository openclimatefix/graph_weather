import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Dict, List

class MaskedReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, masks=None):
        """
        Args:
            predictions: Dict of field predictions, each with shape [E, B, T, H, W] or [B, T, H, W]
            targets: Dict of field targets, each with shape [B, T, H, W]
            masks: Dict of field masks, each with shape [B, T, N] or [B, T, H, W] or None
        """
        total_loss = 0.0
        field_losses = {}
        
        for field, field_preds in predictions.items():
            field_target = targets[field]
            field_mask = masks.get(field, None) if masks is not None else None
            
            # Use the target's spatial dimensions
            B, T, H, W = field_target.shape
            
            # Process the mask based on its dimensions.
            if field_mask is not None:
                if field_mask.dim() == 4:
                    # Already a spatial mask: [B, T, H, W]
                    spatial_mask = field_mask.unsqueeze(0)  # [1, B, T, H, W]
                elif field_mask.dim() == 3:
                    # Patch masks: [B, T, N]. Assume N is a square number.
                    N = field_mask.shape[-1]
                    grid_size = int(math.sqrt(N))
                    spatial_mask = field_mask.view(B, T, grid_size, grid_size)
                    spatial_mask = F.interpolate(spatial_mask.float(), size=(H, W), mode='nearest')
                    spatial_mask = spatial_mask.unsqueeze(0)  # [1, B, T, H, W]
                else:
                    raise ValueError("Unexpected mask dimensions")
            else:
                spatial_mask = torch.ones((1, B, T, H, W), device=field_target.device)
            
            # Expand target if predictions have an ensemble dimension.
            if field_preds.dim() == 5:
                field_target_expanded = field_target.unsqueeze(0).expand_as(field_preds)
            elif field_preds.dim() == 4:
                field_target_expanded = field_target
            else:
                raise ValueError("Unexpected prediction dimensions")
            
            mse = self.mse_loss(field_preds, field_target_expanded)
            masked_mse = (mse * spatial_mask).sum() / (spatial_mask.sum() + 1e-8)
            ensemble_loss = masked_mse
            
            field_losses[field] = ensemble_loss
            total_loss += ensemble_loss
        
        avg_loss = total_loss / len(predictions)
        return avg_loss, field_losses


class PhysicalConsistencyLoss(nn.Module):
    """
    Loss function that enforces physical constraints between different atmospheric variables.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, predictions):
        loss = 0.0
        
        # Enforce non-negative wind speed from u10 and v10.
        if all(field in predictions for field in ['u10', 'v10']):
            u_wind = predictions['u10'].mean(dim=0)
            v_wind = predictions['v10'].mean(dim=0)
            wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
            negative_speed_loss = F.relu(-wind_speed).mean()
            loss += negative_speed_loss
        
        # Penalize excessive temperature gradients in t2m.
        if 't2m' in predictions:
            temp = predictions['t2m'].mean(dim=0)  # [B, T, H, W]
            if temp.shape[-2] > 1 and temp.shape[-1] > 1:
                horiz_grad = (torch.abs(temp[:, :, 1:, :] - temp[:, :, :-1, :]).mean() +
                              torch.abs(temp[:, :, :, 1:] - temp[:, :, :, :-1]).mean())
            else:
                horiz_grad = torch.tensor(0.0, device=temp.device)
            
            if temp.shape[1] > 1:
                temp_grad = torch.abs(temp[:, 1:, :, :] - temp[:, :-1, :, :]).mean()
            else:
                temp_grad = torch.tensor(0.0, device=temp.device)
            
            grad_threshold = 10.0  # degrees per grid cell or time step
            excessive_grad_loss = F.relu(horiz_grad - grad_threshold) + F.relu(temp_grad - grad_threshold)
            loss += excessive_grad_loss
            
        return loss


class AtmoRepLoss(nn.Module):
    def __init__(self, config, field_weights=None):
        super().__init__()
        self.config = config
        self.recon_loss = MaskedReconstructionLoss(config)
        self.phys_loss = PhysicalConsistencyLoss(config)
        
        # Weights for loss components
        self.recon_weight = 1.0
        self.phys_weight = 0.1
        self.field_weights = field_weights or {field: 1.0 for field in config.input_fields}
        
    def forward(self, predictions, targets, masks=None):
        """
        Compute total loss.
        
        Args:
            predictions: Dict of predicted fields, each with shape [E, B, T, H, W] or [B, T, H, W]
            targets: Dict of target fields, each with shape [B, T, H, W]
            masks: Dict of masks for fields, each with shape [B, T, N] or [B, T, H, W] (optional)
        
        Returns:
            total_loss (torch.Tensor): Combined loss.
            loss_details (dict): Breakdown of loss components.
        """
        recon_loss, field_losses = self.recon_loss(predictions, targets, masks)
        phys_loss = self.phys_loss(predictions)
        total_loss = self.recon_weight * recon_loss + self.phys_weight * phys_loss
        
        return total_loss, {
            'reconstruction': recon_loss.item(),
            'physical': phys_loss.item(),
            'field_losses': {k: v.item() for k, v in field_losses.items()}
        }
