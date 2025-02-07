"""
    Low-Rank Adaptation (LoRA) for Linear Layers:
    - Adds trainable rank-constrained adaptations to frozen pre-trained models.
"""
import torch 
import torch.nn as nn
import math
class LoraLayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        """
        Args:
            in_features (int): Input features of the linear layer.
            out_features (int): Output features of the linear layer.
            r (int): Rank of the low-rank approximation.
            alpha (float): Scaling factor for LoRA weights.
        """
        super().__init__()
        self.r = r
        self.alpha = alpha

        # Base linear transformation (frozen during LoRA training)
        self.base_linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_features))

        # Scaling factor for LoRA updates
        self.scaling = self.alpha / self.r

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Forward pass with LoRA adaptation.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        base_output = self.base_linear(x)  # Apply frozen base layer
        lora_output = torch.matmul(x, self.lora_A).matmul(self.lora_B)  # LoRA adjustment
        return base_output + self.scaling * lora_output