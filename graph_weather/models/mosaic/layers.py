"""MOSAIC transformer block and native-grid processor.

The processor keeps the defining property of MOSAIC: spatial interactions
are computed at the native resolution before any coarsening occurs
(arXiv:2604.16429, Section 4.3). Entering the hierarchy at a coarser level
was shown to introduce high-frequency aliasing rather than merely smoothing
the forecast.
"""

from typing import Optional

import torch
from torch import nn

from .block_sparse_attention import BlockSparseAttention
from .coarsen import HealpixCoarsen, HealpixRefine


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Eq. 14)."""

    def __init__(self, dim: int, hidden_ratio: float = 4.0):
        """Initialize the feed-forward network.

        Args:
            dim: Model feature dimension.
            hidden_ratio: Hidden dimension as a multiple of dim.
        """
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.value_proj = nn.Linear(dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Tensor of shape (batch, n_tokens, dim).

        Returns:
            Tensor of shape (batch, n_tokens, dim).
        """
        return self.out_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.value_proj(x))


class MosaicTransformerBlock(nn.Module):
    """Pre-norm transformer block with block-sparse attention (Eq. 14)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 64,
        top_n: int = 3,
        num_kv_heads: Optional[int] = None,
        hidden_ratio: float = 4.0,
    ):
        """Initialize the transformer block.

        Args:
            dim: Model feature dimension.
            num_heads: Number of query heads.
            block_size: Number of tokens per attention block.
            top_n: Number of key blocks selected per query block.
            num_kv_heads: Number of key/value heads for grouped-query
                attention.
            hidden_ratio: Feed-forward hidden dimension multiplier.
        """
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = BlockSparseAttention(
            dim,
            num_heads,
            block_size=block_size,
            top_n=top_n,
            num_kv_heads=num_kv_heads,
        )
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_ratio=hidden_ratio)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention and feed-forward with residual connections.

        Args:
            x: Tensor of shape (batch, n_tokens, dim).
            coords: Optional (n_tokens, 2) tensor of (lat, lon) in radians.

        Returns:
            Tensor of shape (batch, n_tokens, dim).
        """
        x = x + self.attention(self.attention_norm(x), coords)
        return x + self.ffn(self.ffn_norm(x))


class MosaicProcessor(nn.Module):
    """Multi-scale processor operating first at the native resolution.

    The first stage runs block-sparse attention on the input tokens before
    any pooling, which is what preserves fine-scale spectral content
    (Section 4.3). Later stages operate on progressively coarser levels and
    their outputs are added back through skip connections.
    """

    def __init__(
        self,
        dim: int,
        depths: tuple[int, ...] = (2, 2),
        num_heads: int = 4,
        block_size: int = 16,
        top_n: int = 2,
        num_kv_heads: Optional[int] = None,
    ):
        """Initialize the processor.

        Args:
            dim: Model feature dimension, kept constant across stages.
            depths: Number of transformer blocks at each resolution level.
            num_heads: Number of query heads.
            block_size: Number of tokens per attention block.
            top_n: Number of key blocks selected per query block.
            num_kv_heads: Number of key/value heads for grouped-query
                attention.
        """
        super().__init__()
        if len(depths) < 1:
            raise ValueError("depths must contain at least one stage")
        self.depths = depths
        self.stages = nn.ModuleList()
        for depth in depths:
            self.stages.append(
                nn.ModuleList(
                    [
                        MosaicTransformerBlock(
                            dim,
                            num_heads,
                            block_size=block_size,
                            top_n=top_n,
                            num_kv_heads=num_kv_heads,
                        )
                        for _ in range(depth)
                    ]
                )
            )
        n_transitions = len(depths) - 1
        # The processor tracks token features only, so the pooling layers
        # are built without their relative-position term (Eq. 12-13). Supply
        # positions by calling HealpixCoarsen/HealpixRefine directly.
        self.coarsen = nn.ModuleList(
            [HealpixCoarsen(dim, dim, use_positions=False) for _ in range(n_transitions)]
        )
        self.refine = nn.ModuleList(
            [HealpixRefine(dim, dim, use_positions=False) for _ in range(n_transitions)]
        )

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the multi-scale processor.

        Args:
            x: Tensor of shape (batch, n_tokens, dim) in a locality
                preserving order. n_tokens must be divisible by
                4 ** (len(depths) - 1).
            coords: Optional (n_tokens, 2) tensor of (lat, lon) in radians
                used by the native-resolution stage.

        Returns:
            Tensor of shape (batch, n_tokens, dim).
        """
        skips = []
        level_coords = coords
        for index, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, level_coords)
            if index < len(self.coarsen):
                skips.append(x)
                x = self.coarsen[index](x)
                level_coords = None
        for index in reversed(range(len(self.refine))):
            x = self.refine[index](x) + skips[index]
        return x
