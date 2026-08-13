"""
Perceiver Transformer Processor

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
    """
    Architecture configuration for ``PerceiverProcessor``.

    Attributes:
        input_dim: Number of channels of the incoming features. It has to match the width
            produced by the encoder, which `Swin3DEncoder` defaults to 96, so one of the
            two defaults must be overridden for the pair to compose.
        latent_dim: Number of channels of the projected output.
        d_model: Working width of the transformer encoder.
        max_seq_len: Upper bound validated in ``__post_init__``. It is not consulted
            when the encoder is built.
        num_self_attention_layers: Number of transformer encoder layers.
        num_cross_attention_layers: Kept as part of the configuration. This
            implementation has no cross-attention stack and never reads it.
        num_attention_heads: Number of attention heads per layer.
        hidden_dropout: Dropout probability used inside the encoder layers.
        attention_dropout: Validated in ``__post_init__`` but not applied. The
            encoder layers use ``hidden_dropout`` for both attention and
            feed-forward dropout.
        qk_head_dim: Kept as part of the configuration; currently unused. The head
            dimension follows from ``d_model`` and ``num_attention_heads``.
        activation_fn: Name of the activation function used in the feed-forward blocks.
        layer_norm_eps: Kept as part of the configuration; currently unused. The
            encoder layers use the PyTorch default of 1e-5.

    Raises:
        ValueError: If ``input_dim``, ``max_seq_len`` or ``num_attention_heads`` is not
            positive, or if ``hidden_dropout`` or ``attention_dropout`` lies outside [0, 1].
    """

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

    def __post_init__(self):
        # Validate parameters
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if not 0 <= self.hidden_dropout <= 1:
            raise ValueError("hidden_dropout must be between 0 and 1")
        if not 0 <= self.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")


class PerceiverProcessor(nn.Module):
    """
    Process encoded features and pool them into a single latent vector per sample.

    The input is projected to ``d_model``, passed through a stack of self-attention
    transformer encoder layers, projected to ``latent_dim`` and finally averaged over the
    sequence dimension.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize the processor.

        Args:
            config (Optional[ProcessorConfig]): Architecture configuration. A default
                ``ProcessorConfig`` is used when ``None`` is given.
        """
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
        """
        Encode a batch of features and pool it into a latent representation.

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, input_dim).
            attention_mask (Optional[torch.Tensor]): Boolean mask of shape (batch, seq_len).
                Positions that are ``False`` are masked out of the attention.

        Returns:
            torch.Tensor: Pooled latent representation of shape (batch, latent_dim).
        """
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
