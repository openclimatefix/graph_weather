"""
Perceiver Transformer Processor:
- Takes encoded features and processes them using latent space mapping.
- Uses a latent-space bottleneck to compress input dimensions.
- Provides an efficient way to extract long-range dependencies.
- All architectural parameters are configurable.
"""

from dataclasses import dataclass
from typing import Optional

import einops
import torch.nn as nn


@dataclass
class ProcessorConfig:
    input_dim: int = 256  # Match Swin3D output
    latent_dim: int = 512
    d_model: int = 256  # Match input_dim for consistency
    max_seq_len: int = 4096
    num_self_attention_layers: int = 6
    num_cross_attention_layers: int = 2
    num_attention_heads: int = 8
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    qk_head_dim: Optional[int] = 32
    activation_fn: str = "gelu"
    layer_norm_eps: float = 1e-12


class PerceiverProcessor(nn.Module):
    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__()
        self.config = config or ProcessorConfig()

        # Input projection to match d_model
        self.input_projection = nn.Linear(self.config.input_dim, self.config.d_model)

        # Simplified architecture using transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.d_model * 4,
                dropout=self.config.hidden_dropout,
                activation=self.config.activation_fn,
            ),
            num_layers=self.config.num_self_attention_layers,
        )

        # Output projection
        self.output_projection = nn.Linear(self.config.d_model, self.config.latent_dim)

    def forward(self, x, attention_mask=None):
        # Handle 4D input using einops for clearer reshaping
        if len(x.shape) == 4:
            # Rearrange from (batch, seq, height, width) to (batch, seq*height*width, features)
            x = einops.rearrange(x, "b s h w -> b (s h w) c")

        # Project input
        x = self.input_projection(x)

        # Apply transformer encoder with einops for transpose operations
        if attention_mask is not None:
            # Convert boolean mask to float mask where True -> 0, False -> -inf
            mask = ~attention_mask
            mask = mask.float().masked_fill(mask, float("-inf"))
            x = einops.rearrange(
                x, "b s c -> s b c"
            )  # (batch, seq, channels) -> (seq, batch, channels)
            x = self.encoder(x, src_key_padding_mask=mask)
            x = einops.rearrange(
                x, "s b c -> b s c"
            )  # (seq, batch, channels) -> (batch, seq, channels)
        else:
            x = einops.rearrange(x, "b s c -> s b c")
            x = self.encoder(x)
            x = einops.rearrange(x, "s b c -> b s c")

        # Project to latent dimension and pool
        x = self.output_projection(x)
        x = x.mean(dim=1)  # Global average pooling

        return x
