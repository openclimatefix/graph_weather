import torch.nn as nn
from ..config import AtmoRepConfig
from .transformer import TransformerBlock

class Decoder(nn.Module):
    """
    A Decoder module designed to process the features of a given field, 
    using transformer blocks with U-Net-like skip connections for efficient 
    multi-resolution feature propagation.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters such as hidden_dim, 
                                 num_layers, etc.
    """
    def __init__(self, config: AtmoRepConfig):
        super().__init__()
        self.config = config
        
        # Define the transformer blocks with U-Net-like connections
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers // 2)
        ])
        
        # Define projections for skip connections (used in U-Net-like structure)
        self.skip_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(len(self.blocks))
        ])
        
        # Layer normalization after all transformer blocks
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x, skip_features):
        """
        Forward pass through the Decoder. The decoder applies transformer blocks and uses skip 
        connections from lower resolution features (similar to a U-Net architecture).

        Args:
            x (torch.Tensor): Input tensor, shape [B, T, N, D] where B is the batch size, 
                              T is the number of time steps, N is the number of patches, and D is the feature dimension.
            skip_features (list of torch.Tensor): A list of features from previous layers (multi-resolution 
                                                   features), used for skip connections. Each entry should 
                                                   have the same shape as the input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the transformer blocks and layer normalization.
        """
        # Process with transformer blocks and U-Net-like connections
        for i, block in enumerate(self.blocks):
            # Use skip connections for multi-resolution features
            if i < len(skip_features):
                # Select the skip feature corresponding to the current resolution
                skip_idx = len(skip_features) - 1 - i
                skip = self.skip_projections[i](skip_features[skip_idx])
                # Apply the transformer block with the skip connection
                x = block(x + skip)
            else:
                # If no skip feature, just apply the transformer block
                x = block(x)
        
        # Apply final layer normalization
        x = self.norm(x)
        return x
