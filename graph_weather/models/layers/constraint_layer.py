import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalConstraintLayer(nn.Module):
    """
    This module implements several constraint types on the network’s intermediate outputs ỹ,
    given the corresponding low-resolution input x. The following equations are implemented
    (with all operations acting per patch – here, a patch is the full grid of H×W pixels):
    
    Additive constraint:
      y = ỹ + x - avg(ỹ)
      
    Multiplicative constraint:
      y = ỹ * ( x / avg(ỹ) )
      
    Softmax constraint:
      y = exp(ỹ) * ( x / sum(exp(ỹ)) )
    
    We assume that both the intermediate outputs and the low-resolution reference are 4D
    tensors in grid format, with shape [B, C, H, W], where n = H*W is the number of pixels
    (or nodes) in a patch.
    """
    def __init__(self, grid_shape, constraint_type='additive', exp_factor=1.0):
        super().__init__()
        self.constraint_type = constraint_type
        self.grid_shape = grid_shape
        self.exp_factor = exp_factor
        
    def forward(self, hr_graph, lr_graph):
        """
        Args:
            hr_output: High-resolution model output [B, C, H, W]
            lr_input: Low-resolution input [B, C, h, w]
        """ 
        # Check if inputs are in graph (3D) or grid (4D) formats.
        if hr_graph.dim() == 3:
        # Convert graph outputs to grid format
            batch_size, _, features = hr_graph.shape
            hr_grid = hr_graph.view(batch_size, features, 
                                self.grid_shape[0], self.grid_shape[1])  # [B, C, H, W]
            lr_grid = lr_graph.view(batch_size, features,
                                self.grid_shape[0], self.grid_shape[1])  # [B, C, H, W]
        elif hr_graph.dim() == 4:
            # Already in grid format: [B, C, H, W]
            batch_size, features, H, W = hr_graph.shape
            if (H, W) != self.grid_shape:
                raise ValueError(f"Expected spatial dimensions {self.grid_shape}, got {(H, W)}")
            hr_grid = hr_graph
            lr_grid = lr_graph
        else:
            raise ValueError("Input tensor must be either 3D (graph) or 4D (grid).")

        if self.constraint_type == 'additive':
            result = self.additive_constraint(hr_grid, lr_grid)
        elif self.constraint_type == 'multiplicative':
            result = self.multiplicative_constraint(hr_grid, lr_grid)
        elif self.constraint_type == 'softmax':
            result = self.softmax_constraint(hr_grid, lr_grid)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
        
        # Convert back to graph format
        return result.view(batch_size, -1, features)
    
    def additive_constraint(self, hr, lr):
        """
        Enforces local conservation on the high-resolution field hr so that when it is pooled
        the result matches the low-resolution reference lr.
        
        hr: high-resolution tensor [B, C, H_hr, W_hr]
        lr: low-resolution tensor [B, C, H_lr, W_lr]
        """
        # Downscale hr so that its spatial dimensions match those of lr.
        lr_downscaled = F.adaptive_avg_pool2d(hr, output_size=lr.shape[-2:])
        # Compute the local delta needed to correct hr:
        delta = lr - lr_downscaled
        # Upsample delta to hr resolution using nearest-neighbor (to replicate values exactly)
        delta_up = F.interpolate(delta, size=hr.shape[-2:], mode='nearest')
        # Apply local correction:
        hr_adjusted = hr + delta_up
        return hr_adjusted

    def multiplicative_constraint(self, hr, lr):
        # y = ỹ * ( x / avg(ỹ) )
        avg_hr = hr.mean(dim=(-2, -1), keepdim=True)
        return hr * (lr / (avg_hr + 1e-8))

    def softmax_constraint(self, hr, lr):
        # y = exp(ỹ) * ( x / sum(exp(ỹ)) )
        B, C, H, W = hr.shape
        hr_flat = hr.view(B, C, -1)
        # Optionally, scale hr before exponentiation:
        exp_hr = torch.exp(self.exp_factor * hr_flat)
        sum_exp = exp_hr.sum(dim=-1, keepdim=True)
        lr_flat = lr.view(B, C, -1)
        y_flat = exp_hr * (lr_flat / (sum_exp + 1e-8))
        return y_flat.view(B, C, H, W)
