"""Module for physical constraint layers used in graph weather models.

This module implements several constraints on a network’s intermediate outputs,
ensuring physical consistency with an input at a lower resolution.

"""

import torch
import torch.nn as nn


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

    def __init__(
        self, model, grid_shape, upsampling_factor, constraint_type="none", exp_factor=1.0
    ):
        """Initialize the PhysicalConstraintLayer.

        Args:
            model (nn.Module): The model containing the helper methods
                'graph_to_grid' and 'grid_to_graph'.
            grid_shape (tuple): Expected spatial dimensions (H, W) of the
                high-resolution grid.
            upsampling_factor (int): Factor by which the low-resolution grid is upsampled.
            constraint_type (str, optional): The constraint to apply. Options are
                'additive', 'multiplicative', or 'softmax'. Defaults to "none".
            exp_factor (float, optional): Exponent factor for the softmax constraint.
                Defaults to 1.0.
        """
        super().__init__()
        self.model = model
        self.constraint_type = constraint_type
        self.grid_shape = grid_shape
        self.exp_factor = exp_factor
        self.upsampling_factor = upsampling_factor
        self.pool = nn.AvgPool2d(kernel_size=upsampling_factor)

    def forward(self, hr_graph, lr_graph):
        """Apply the selected physical constraint.

        Processes the high-resolution output and low-resolution input by converting
        between graph and grid formats as needed, and then applying the specified constraint.

        Args:
            hr_graph (torch.Tensor): High-resolution model output in either graph (3D)
                or grid (4D) format.
            lr_graph (torch.Tensor): Low-resolution input in the corresponding 
                graph or grid format.

        Returns:
            torch.Tensor: The adjusted output in graph format.
        """
        # Check if inputs are in graph (3D) or grid (4D) formats.
        if hr_graph.dim() == 3:
            # Convert graph format to grid format
            hr_grid = self.model.graph_to_grid(hr_graph)
            lr_grid = self.model.graph_to_grid(lr_graph)
        elif hr_graph.dim() == 4:
            # Already in grid format: [B, C, H, W]
            _, _, H, W = hr_graph.shape
            if (H, W) != self.grid_shape:
                raise ValueError(f"Expected spatial dimensions {self.grid_shape}, got {(H, W)}")
            hr_grid = hr_graph
            lr_grid = lr_graph
        else:
            raise ValueError("Input tensor must be either 3D (graph) or 4D (grid).")

        # Apply constraint based on type in grid format
        if self.constraint_type == "additive":
            result = self.additive_constraint(hr_grid, lr_grid)
        elif self.constraint_type == "multiplicative":
            result = self.multiplicative_constraint(hr_grid, lr_grid)
        elif self.constraint_type == "softmax":
            result = self.softmax_constraint(hr_grid, lr_grid)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")

        # Convert grid back to graph format
        return self.model.grid_to_graph(result)

    def additive_constraint(self, hr, lr):
        """Enforces local conservation using an additive correction:
        y = ỹ + ( x - avg(ỹ) )
        where avg(ỹ) is computed per patch (via an average-pooling layer).

        For the additive constraint we follow the paper’s formulation using a Kronecker
        product to expand the discrepancy between the low-resolution field and the
        average of the high-resolution output.

        hr: high-resolution tensor [B, C, H_hr, W_hr]
        lr: low-resolution tensor [B, C, h_lr, w_lr]
        (with H_hr = upsampling_factor * h_lr & W_hr = upsampling_factor * w_lr)
        """
        # Convert grids to graph format using model's mapping
        hr_graph = self.model.grid_to_graph(hr)
        lr_graph = self.model.grid_to_graph(lr)

        # Apply constraint logic
        # Compute average over NODES
        avg_hr = hr_graph.mean(dim=1, keepdim=True)
        diff = lr_graph - avg_hr

        # Expand difference using spatial mapping
        diff_expanded = diff.repeat(1, self.upsampling_factor**2, 1)

        # Apply correction and convert back to GRID format
        adjusted_graph = hr_graph + diff_expanded
        return self.model.graph_to_grid(adjusted_graph)

    def multiplicative_constraint(self, hr, lr):
        """Enforce conservation using a multiplicative correction in graph space.

        The correction is applied by scaling the high-resolution output by a ratio computed
        from the low-resolution input and the average of the high-resolution output.

        Args:
            hr (torch.Tensor): High-resolution tensor in grid format [B, C, H_hr, W_hr].
            lr (torch.Tensor): Low-resolution tensor in grid format [B, C, h_lr, w_lr].

        Returns:
            torch.Tensor: Adjusted high-resolution tensor in grid format.
        """
        # Convert grids to graph format using model's mapping
        hr_graph = self.model.grid_to_graph(hr)
        lr_graph = self.model.grid_to_graph(lr)

        # Apply constraint logic
        # Compute average over NODES
        avg_hr = hr_graph.mean(dim=1, keepdim=True)
        lr_patch_avg = lr_graph.mean(dim=1, keepdim=True)

        # Compute ratio and expand to match HR graph structure
        ratio = lr_patch_avg / (avg_hr + 1e-8)

        # Apply multiplicative correction and convert back to GRID format
        adjusted_graph = hr_graph * ratio
        return self.model.graph_to_grid(adjusted_graph)

    def softmax_constraint(self, y, lr):
        """Apply a softmax-based constraint correction.

        The softmax correction scales the exponentiated high-resolution output so that the
        sum over spatial blocks matches the low-resolution reference.

        Args:
            y (torch.Tensor): High-resolution tensor in grid format [B, C, H, W].
            lr (torch.Tensor): Low-resolution tensor in grid format [B, C, h, w].

        Returns:
            torch.Tensor: Adjusted high-resolution tensor in grid format after applying
            the softmax constraint.
        """
        # Apply the exponential function
        y = torch.exp(self.exp_factor * y)

        # Pool over spatial blocks
        kernel_area = self.upsampling_factor**2
        sum_y = self.pool(y) * kernel_area

        # Ensure that lr * (1/sum_y) is contiguous
        ratio = (lr * (1 / sum_y)).contiguous()

        # Use device of lr for kron expansion:
        device = lr.device
        expansion = torch.ones((self.upsampling_factor, self.upsampling_factor),
            device=device)

        # Expand the low-resolution ratio and correct the y values so that the block sum matches lr.
        out = y * torch.kron(ratio, expansion)
        return out
