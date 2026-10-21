"""Learnable HEALPix pooling and unpooling (MOSAIC Eq. 12-13).

In NESTED ordering the four children of a HEALPix pixel are consecutive, so
coarsening is a reshape followed by a learnable projection of the stacked
child features and their relative positions (arXiv:2604.16429, Section 4.3).
Refinement mirrors it. No graph library is needed.
"""

from typing import Optional

import torch
from torch import nn


class HealpixCoarsen(nn.Module):
    """Pool four sibling pixels into their parent (Eq. 12)."""

    def __init__(self, in_dim: int, out_dim: int, factor: int = 4, use_positions: bool = True):
        """Initialize the pooling layer.

        Args:
            in_dim: Feature dimension of the fine level.
            out_dim: Feature dimension of the coarse level.
            factor: Number of children per parent; 4 for HEALPix.
            use_positions: Whether to include the relative-position term of
                Eq. 12. Set to False when no positions will be supplied so
                the layer holds no unused parameters.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor = factor
        self.feature_proj = nn.Linear(factor * in_dim, out_dim, bias=False)
        self.position_proj = nn.Linear(factor * 3, out_dim, bias=False) if use_positions else None
        self.norm = nn.RMSNorm(out_dim)

    def forward(self, x: torch.Tensor, rel_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Coarsen features by a factor of four.

        Args:
            x: Tensor of shape (batch, n_tokens, in_dim); n_tokens must be
                divisible by the pooling factor.
            rel_pos: Optional (n_tokens, 3) child positions relative to the
                parent pixel centre.

        Returns:
            Tensor of shape (batch, n_tokens // factor, out_dim).
        """
        batch, n_tokens, _ = x.shape
        if n_tokens % self.factor != 0:
            raise ValueError(f"n_tokens {n_tokens} not divisible by factor {self.factor}")
        n_parent = n_tokens // self.factor
        grouped = x.reshape(batch, n_parent, self.factor * self.in_dim)
        out = self.feature_proj(grouped)
        if self.position_proj is None:
            if rel_pos is not None:
                raise ValueError(
                    "rel_pos was given but the layer was built with use_positions=False"
                )
        else:
            if rel_pos is None:
                raise ValueError(
                    "rel_pos is required when use_positions=True; pass positions or "
                    "build the layer with use_positions=False"
                )
            pos = rel_pos.reshape(n_parent, self.factor * 3)
            out = out + self.position_proj(pos).unsqueeze(0)
        return self.norm(out)


class HealpixRefine(nn.Module):
    """Expand a parent pixel back into its four children (Eq. 13)."""

    def __init__(self, in_dim: int, out_dim: int, factor: int = 4, use_positions: bool = True):
        """Initialize the unpooling layer.

        Args:
            in_dim: Feature dimension of the coarse level.
            out_dim: Feature dimension of the fine level.
            factor: Number of children per parent; 4 for HEALPix.
            use_positions: Whether to include the relative-position term of
                Eq. 13. Set to False when no positions will be supplied so
                the layer holds no unused parameters.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor = factor
        self.feature_proj = nn.Linear(in_dim, factor * out_dim, bias=False)
        self.position_proj = nn.Linear(3, out_dim, bias=False) if use_positions else None
        self.norm = nn.RMSNorm(out_dim)

    def forward(self, x: torch.Tensor, rel_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Refine features by a factor of four.

        Args:
            x: Tensor of shape (batch, n_parent, in_dim).
            rel_pos: Optional (n_parent * factor, 3) child positions
                relative to the parent pixel centre.

        Returns:
            Tensor of shape (batch, n_parent * factor, out_dim).
        """
        batch, n_parent, _ = x.shape
        out = self.feature_proj(x).reshape(batch, n_parent * self.factor, self.out_dim)
        if self.position_proj is None:
            if rel_pos is not None:
                raise ValueError(
                    "rel_pos was given but the layer was built with use_positions=False"
                )
        else:
            if rel_pos is None:
                raise ValueError(
                    "rel_pos is required when use_positions=True; pass positions or "
                    "build the layer with use_positions=False"
                )
            out = out + self.position_proj(rel_pos).unsqueeze(0)
        return self.norm(out)
