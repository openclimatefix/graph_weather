from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from .transformer import TransformerBlock

class FieldVisionTransformer(nn.Module):
    """
    A Vision Transformer (ViT) designed to process individual fields in a model.
    It extracts features from spatial patches across multiple time steps using
    a transformer architecture.

    Args:
        hidden_dim (int): Dimensionality of the transformer embeddings.
        patch_size (int): Size of each spatial patch (square).
        spatial_dims (tuple[int, int]): Full spatial dimensions (height, width).
        time_steps (int): Number of temporal steps in the input.
        num_layers (int): Number of transformer blocks.
        field_name (str): Name of the field this transformer is processing (e.g., 'temperature').
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        spatial_dims: tuple[int, int],
        time_steps: int,
        num_layers: int,
        field_name: str = "unknown_field",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.field_name = field_name

        # Calculate how many patches we'll get from the spatial dimensions
        height, width = spatial_dims
        self.num_patches = (height // patch_size) * (width // patch_size)

        # Patch embedding layer: transforms each patch into a hidden_dim embedding
        self.patch_embed = nn.Conv2d(
            in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings for spatial patches + separate embeddings for time steps
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, time_steps, hidden_dim))

        # Instantiate the transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim=hidden_dim) for _ in range(num_layers)]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize weights for layers and embeddings
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initializes the weights of the patch embedding and position/time embeddings.
        """
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the FieldVisionTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H, W], where:
                - B is batch size
                - T is number of time steps
                - H, W are spatial dimensions
            mask (torch.Tensor, optional): A mask tensor of shape [B, T, N], where
                N is the number of patches. If provided, masked patches will be replaced
                by a learnable mask token.

        Returns:
            (torch.Tensor, list):
                - The final output after the transformer blocks (shape [B, T, N, hidden_dim]).
                - A list of intermediate features for U-Net-like skip connections.
        """
        B, T, H, W = x.shape
        if T != self.time_steps:
            raise ValueError(f"Expected input with {self.time_steps} time steps, but got {T}.")

        # === Extract patches for each time step ===
        tokens_list = []
        for t in range(T):
            # x[:, t] => shape [B, H, W]
            # Unsqueeze to make it [B, 1, H, W] for Conv2d
            frame = x[:, t].unsqueeze(1)
            patch_tokens = self.patch_embed(frame)  # [B, hidden_dim, H//patch_size, W//patch_size]
            # Flatten spatial dims => [B, (H//patch_size)*(W//patch_size), hidden_dim]
            patch_tokens = rearrange(patch_tokens, "b d hp wp -> b (hp wp) d")
            tokens_list.append(patch_tokens)

        # Stack along the time dimension => shape [B, T, N, hidden_dim]
        tokens = torch.stack(tokens_list, dim=1)

        # === Add positional and time embeddings ===
        # pos_embed: shape [1, N, hidden_dim]
        # time_embed: shape [1, T, hidden_dim]
        # Expand them to match tokens shape
        tokens = tokens + self.pos_embed.unsqueeze(1)  # [B, T, N, hidden_dim]
        tokens = tokens + self.time_embed.unsqueeze(2)  # [B, T, N, hidden_dim]

        # === Apply optional mask ===
        if mask is not None:
            # mask: shape [B, T, N]
            # Expand mask to match tokens => [B, T, N, hidden_dim]
            mask_expanded = mask.unsqueeze(-1).expand_as(tokens)
            # Learnable mask token
            mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))
            # Replace masked patches
            tokens = tokens * (1 - mask_expanded) + mask_token * mask_expanded

        # === Pass through transformer blocks ===
        # We'll store intermediate features for U-Net-like connections.
        features = []
        x_out = tokens
        for i, block in enumerate(self.blocks):
            x_out = block(x_out)
            # Optionally store intermediate feature maps
            # Here, for example, we store every block's output.
            features.append(x_out)

        # Final layer normalization
        x_out = self.norm(x_out)

        return x_out, features