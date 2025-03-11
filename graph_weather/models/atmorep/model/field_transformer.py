import torch
import torch.nn as nn
from einops import rearrange

from ..config import AtmoRepConfig
from .transformer import TransformerBlock


class FieldVisionTransformer(nn.Module):
    """
    A Vision Transformer (ViT) designed to process individual fields in the AtmoRep model.
    This model extracts features from spatial patches and temporal steps using a transformer architecture.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters such as hidden_dim,
                                 patch_size, num_layers, etc.
        field_name (str): Name of the field this transformer is processing (e.g., temperature, humidity).
    """

    def __init__(self, config: AtmoRepConfig, field_name: str):
        super().__init__()
        self.config = config
        self.field_name = field_name

        # Calculate patch and number of patches
        self.patch_dim = config.patch_size**2
        self.num_patches = (config.spatial_dims[0] // config.patch_size) * (
            config.spatial_dims[1] // config.patch_size
        )

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            1, config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size
        )

        # Position embeddings for spatial patches and temporal embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, config.time_steps, config.hidden_dim))

        # Transformer blocks (e.g., attention layers)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # Layer normalization to stabilize the training process
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Initialize weights for layers and embeddings
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the patch embedding and position/time embeddings.
        """
        # Xavier initialization for patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)

        # Normal distribution initialization for position and time embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def forward(self, x, mask=None):
        """
        Forward pass through the Field Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, H, W], where B is the batch size,
                              T is the number of time steps, H and W are spatial dimensions.
            mask (torch.Tensor, optional): Mask tensor of shape [B, T, N] where N is the number of patches,
                                           used to mask certain input tokens.

        Returns:
            torch.Tensor: The transformed tensor after applying transformer blocks and layer normalization.
            list: A list of multi-resolution features from the transformer blocks for U-Net like connections.
        """
        B, T, H, W = x.shape

        # Process each time step separately to extract patch tokens
        tokens = []
        for t in range(T):
            # Apply patch embedding to each time step's image
            patch_tokens = self.patch_embed(x[:, t].unsqueeze(1))  # [B, D, H//P, W//P]
            # Flatten the spatial dimensions (height and width)
            patch_tokens = rearrange(patch_tokens, "b d h w -> b (h w) d")  # [B, N, D]
            tokens.append(patch_tokens)

        # Stack the time tokens into a single tensor: [B, T, N, D]
        tokens = torch.stack(tokens, dim=1)

        # Add position and time embeddings to the tokens
        tokens = tokens + self.pos_embed.unsqueeze(1)  # Add position embeddings
        tokens = tokens + self.time_embed.unsqueeze(2)  # Add time embeddings

        # Apply mask if provided
        if mask is not None:
            mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.config.hidden_dim))
            mask = mask.unsqueeze(-1).expand_as(tokens)  # Expand mask to match tokens shape
            tokens = tokens * (1 - mask) + mask_token * mask  # Apply mask to tokens

        # Pass through the transformer blocks
        features = []
        x = tokens
        for i, block in enumerate(self.blocks):
            x = block(x)  # Apply transformer block
            if i % 3 == 2:  # Store features at every 3rd block (for multi-resolution features)
                features.append(x)

        # Apply final layer normalization
        x = self.norm(x)

        return x, features
