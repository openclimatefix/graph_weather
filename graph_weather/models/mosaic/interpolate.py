"""Cross-attention interpolation between point sets (MOSAIC Eq. 6-7).

MOSAIC moves features between the native input grid and the HEALPix grid
with a cross-attention interpolation layer: queries are formed from the
L2-normalised relative position between a target point and its nearest
source points, while keys and values come from the source features
(arXiv:2604.16429, Section 4.1). Because it is defined purely in terms of
neighbour positions, the same layer works for arbitrary point sets in
either direction.
"""

from typing import Optional

import torch
from torch import nn


def knn_indices(
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    k: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Find the k nearest source points for each target point.

    The pairwise distances are computed in chunks over the target points so
    the full (n_target, n_source) matrix is never held at once.

    Args:
        source_coords: Tensor of shape (n_source, 3) of unit-sphere
            Cartesian coordinates.
        target_coords: Tensor of shape (n_target, 3) of unit-sphere
            Cartesian coordinates.
        k: Number of neighbours per target point.
        chunk_size: Number of target points handled per chunk.

    Returns:
        Long tensor of shape (n_target, k) of source indices.
    """
    k = min(k, source_coords.shape[0])
    chunks = []
    for start in range(0, target_coords.shape[0], chunk_size):
        block = target_coords[start : start + chunk_size]
        distances = torch.cdist(block, source_coords)
        chunks.append(distances.topk(k, dim=-1, largest=False).indices)
    return torch.cat(chunks, dim=0)


class CrossAttentionInterpolator(nn.Module):
    """Interpolate features from one point set to another (Eq. 6-7)."""

    def __init__(self, dim: int, num_neighbors: int = 4):
        """Initialize the interpolator.

        Args:
            dim: Feature dimension of source and target features.
            num_neighbors: Number of source neighbours attended per target.
        """
        super().__init__()
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.q_proj = nn.Linear(3, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        source_feats: torch.Tensor,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate source features onto the target points.

        Args:
            source_feats: Tensor of shape (batch, n_source, dim).
            source_coords: Tensor of shape (n_source, 3) of unit-sphere
                Cartesian coordinates.
            target_coords: Tensor of shape (n_target, 3) of unit-sphere
                Cartesian coordinates.
            neighbor_idx: Optional precomputed (n_target, k) neighbour
                indices; computed on the fly when omitted.

        Returns:
            Tensor of shape (batch, n_target, dim).
        """
        if neighbor_idx is None:
            neighbor_idx = knn_indices(source_coords, target_coords, self.num_neighbors)
        batch = source_feats.shape[0]
        n_target, k = neighbor_idx.shape

        # Relative positions target -> neighbouring source points (Eq. 6).
        rel = source_coords[neighbor_idx] - target_coords.unsqueeze(1)
        # A target that coincides with a source point gives a zero vector,
        # where the direction is undefined and the derivative of the
        # normalisation diverges. Coordinates lie on the unit sphere, so
        # softening the length with 1e-6 under the square root bounds the
        # derivative without affecting well-separated points. Clamping the
        # norm after the fact would not: the derivative would still pass
        # through norm() at zero.
        rel = rel * (rel.pow(2).sum(-1, keepdim=True) + 1e-6).rsqrt()
        queries = self.q_proj(rel)

        feats = self.norm(source_feats)
        gathered = feats[:, neighbor_idx.reshape(-1)].view(batch, n_target, k, -1)
        keys = self.k_proj(gathered)
        values = self.v_proj(gathered)

        scale = self.dim**-0.5
        scores = (queries.unsqueeze(0) * keys).sum(dim=-1) * scale
        weights = scores.softmax(dim=-1)
        out = (weights.unsqueeze(-1) * values).sum(dim=2)
        return self.out_proj(out)
