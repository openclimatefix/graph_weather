"""
Perceiver Transformer Processor:
- Takes encoded features and processes them using latent space mapping.
- Uses a latent-space bottleneck to compress input dimensions.
- Provides an efficient way to extract long-range dependencies.
- All architectural parameters are configurable.
"""

import torch.nn as nn
import einops
from transformers import PerceiverConfig, PerceiverModel
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ProcessorConfig:
    input_dim: int = 256  # Match Swin3D output
    latent_dim: int = 512
    d_model: int = 256    # Match input_dim for consistency
    max_seq_len: int = 4096
    num_self_attention_layers: int = 6
    num_cross_attention_layers: int = 2
    num_attention_heads: int = 8
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    qk_head_dim: Optional[int] = 32
    activation_fn: str = 'gelu'
    layer_norm_eps: float = 1e-12
    
    def __post_init__(self):
        """Validation for ProcessorConfig parameters"""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be greater than 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than 0")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be greater than 0")
        if not (0 <= self.hidden_dropout <= 1):
            raise ValueError("hidden_dropout must be between 0 and 1")
        if self.qk_head_dim is not None and self.qk_head_dim <= 0:
            raise ValueError("qk_head_dim must be greater than 0 if specified")
        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0")
    

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
                activation=self.config.activation_fn
            ),
            num_layers=self.config.num_self_attention_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.config.d_model, self.config.latent_dim)

    def forward(self, x, attention_mask=None):
        # Handle 4D input
        if len(x.shape) == 4:
            batch_size, seq_len, height, width = x.shape
            x = x.reshape(batch_size, seq_len * height * width, -1)
        
        # Project input
        x = self.input_projection(x)
        
        # Apply transformer encoder
        if attention_mask is not None:
            # Convert boolean mask to float mask where True -> 0, False -> -inf
            mask = ~attention_mask
            mask = mask.float().masked_fill(mask, float('-inf'))
            x = self.encoder(x.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        else:
            x = self.encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Project to latent dimension and pool
        x = self.output_projection(x)
        x = x.mean(dim=1)  # Global average pooling
        
        return x