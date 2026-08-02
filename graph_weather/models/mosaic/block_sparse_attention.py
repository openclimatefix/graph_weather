"""Block-sparse attention (BSA) from MOSAIC.

Implements the three-branch block-sparse attention of
"(Sparse) Attention to the Details: Preserving Spectral Fidelity in
ML-based Weather Forecasting Models" (arXiv:2604.16429):

- Compression: queries, keys and values are mean-pooled per block (Eq. 8)
  and dense attention is computed between block representations (Eq. 9),
  broadcast back to all tokens of the query block (Eq. 10).
- Fine-grained selection: each query block (not each token) selects the
  top-n key blocks using the compression attention scores and attends to
  their tokens at full resolution (Eq. 11).
- Local: full attention within each block independently. The paper uses
  block attention instead of a sliding window because a sliding window
  would require handling irregular boundaries on the sphere (Section 4.2).

The three branch outputs are combined with learnable linear gating
functions (Eq. 2). The paper does not specify the gate nonlinearity; this
implementation uses a per-token sigmoid gate per branch, following the NSA
convention the paper builds on.

Tokens must already be ordered so that contiguous index blocks correspond
to spatial neighbourhoods (e.g. HEALPix NESTED ordering, see
graph_weather.models.mosaic.healpix). The module itself is agnostic to
how that ordering was produced, so it works on arbitrary point sets.

This is an independent implementation from the paper text only; no code
from the (unlicensed) reference repository was used.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RotaryEmbedding2D(nn.Module):
    """2D axial rotary position embedding on (lat, lon).

    The head dimension is split in half; the first half is rotated with the
    latitude angle and the second half with the longitude angle
    (arXiv:2604.16429, Section 4.3).
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        """Initialize the rotary embedding.

        Args:
            head_dim: Per-head feature dimension. Must be divisible by 4.
            theta: Base frequency for the rotary embedding.
        """
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
        self.head_dim = head_dim
        quarter = head_dim // 4
        inv_freq = theta ** (-torch.arange(quarter, dtype=torch.float32) / quarter)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _angles(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute rotation angles for each token.

        Args:
            coords: Tensor of shape (N, 2) with (lat, lon) in radians.

        Returns:
            Tensor of shape (N, head_dim // 2) of rotation angles.
        """
        lat = coords[:, 0:1] * self.inv_freq
        lon = coords[:, 1:2] * self.inv_freq
        return torch.cat([lat, lon], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys.

        Args:
            q: Queries of shape (..., N, head_dim).
            k: Keys of shape (..., N, head_dim).
            coords: Tensor of shape (N, 2) with (lat, lon) in radians.

        Returns:
            Rotated (q, k) with unchanged shapes.
        """
        angles = self._angles(coords.to(q.dtype))
        cos = angles.cos().repeat_interleave(2, dim=-1)
        sin = angles.sin().repeat_interleave(2, dim=-1)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            return x * cos + rotated * sin

        return rotate(q), rotate(k)


class BlockSparseAttention(nn.Module):
    """Three-branch block-sparse attention (arXiv:2604.16429, Section 4.2).

    Input tokens are grouped into contiguous blocks of block_size; the
    caller must provide tokens in a locality-preserving order. The sequence
    is padded internally so its length is a multiple of block_size and
    padded positions are masked out of every softmax.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 64,
        top_n: int = 3,
        num_kv_heads: Optional[int] = None,
        use_rope: bool = True,
    ):
        """Initialize block-sparse attention.

        Args:
            dim: Model feature dimension.
            num_heads: Number of query heads.
            block_size: Number of tokens per block.
            top_n: Number of key blocks each query block selects.
            num_kv_heads: Number of key/value heads for grouped-query
                attention. Defaults to num_heads (no grouping).
            use_rope: Whether to apply 2D axial RoPE when coords are given.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.top_n = top_n

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        # One sigmoid gate per branch (compression, selection, local), Eq. 2.
        self.gate = nn.Linear(dim, 3)
        self.rope = RotaryEmbedding2D(self.head_dim) if use_rope else None

    def _pad(
        self, x: torch.Tensor, coords: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
        """Pad tokens so the sequence length is a multiple of block_size."""
        batch, n_tokens, _ = x.shape
        block = self.block_size
        pad = (block - n_tokens % block) % block
        valid = torch.ones(batch, n_tokens + pad, dtype=torch.bool, device=x.device)
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            valid[:, n_tokens:] = False
            if coords is not None:
                coords = F.pad(coords, (0, 0, 0, pad))
        return x, coords, valid, pad

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply block-sparse attention.

        Args:
            x: Token features of shape (batch, n_tokens, dim), ordered so
                that contiguous blocks are spatial neighbourhoods.
            coords: Optional (n_tokens, 2) tensor of (lat, lon) in radians
                used for 2D rotary embeddings.

        Returns:
            Tensor of shape (batch, n_tokens, dim).
        """
        batch, n_tokens, _ = x.shape
        x_p, coords_p, valid, pad = self._pad(x, coords)
        n_padded = x_p.shape[1]
        block = self.block_size
        m = n_padded // block
        heads = self.num_heads
        kv_heads = self.num_kv_heads
        dh = self.head_dim
        group = heads // kv_heads

        q = self.q_proj(x_p).view(batch, n_padded, heads, dh).transpose(1, 2)
        k = self.k_proj(x_p).view(batch, n_padded, kv_heads, dh).transpose(1, 2)
        v = self.v_proj(x_p).view(batch, n_padded, kv_heads, dh).transpose(1, 2)
        if self.rope is not None and coords_p is not None:
            q, k = self.rope(q, k, coords_p)
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        q_b = q.view(batch, heads, m, block, dh)
        k_b = k.view(batch, heads, m, block, dh)
        v_b = v.view(batch, heads, m, block, dh)
        valid_b = valid.view(batch, 1, m, block)
        block_sizes = valid_b.sum(dim=-1, keepdim=True).clamp(min=1)

        # Compression branch (Eq. 8-10): masked mean pooling per block.
        mean_mask = valid_b.unsqueeze(-1).to(q_b.dtype)
        q_bar = (q_b * mean_mask).sum(dim=3) / block_sizes
        k_bar = (k_b * mean_mask).sum(dim=3) / block_sizes
        v_bar = (v_b * mean_mask).sum(dim=3) / block_sizes
        scale = dh**-0.5
        scores = torch.einsum("bhid,bhjd->bhij", q_bar, k_bar) * scale
        # Fully-padded key blocks are masked before the softmax and before
        # the top-n so padding can never be selected.
        block_valid = valid_b.any(dim=-1)
        scores = scores.masked_fill(~block_valid.unsqueeze(2), float("-inf"))
        attn_cg = scores.softmax(dim=-1)
        o_cg = torch.einsum("bhij,bhjd->bhid", attn_cg, v_bar)
        o_cg = o_cg.unsqueeze(3).expand(batch, heads, m, block, dh)

        # Fine-grained selection branch (Eq. 11): the top-n key blocks are
        # chosen per query block and shared by all tokens in that block.
        top_n = min(self.top_n, m)
        sel_scores = scores.mean(dim=1)
        sel_idx = sel_scores.topk(top_n, dim=-1).indices
        sel_idx_e = sel_idx.view(batch, 1, m, top_n, 1, 1).expand(batch, heads, m, top_n, block, dh)
        k_sel = k_b.unsqueeze(2).expand(batch, heads, m, m, block, dh)
        k_sel = torch.gather(k_sel, 3, sel_idx_e).flatten(3, 4)
        v_sel = v_b.unsqueeze(2).expand(batch, heads, m, m, block, dh)
        v_sel = torch.gather(v_sel, 3, sel_idx_e).flatten(3, 4)
        valid_sel = valid_b.unsqueeze(2).expand(batch, 1, m, m, block)
        idx_mask = sel_idx.view(batch, 1, m, top_n, 1).expand(batch, 1, m, top_n, block)
        valid_sel = torch.gather(valid_sel, 3, idx_mask).flatten(3, 4)
        fg_scores = torch.einsum("bhitd,bhisd->bhits", q_b, k_sel) * scale
        fg_scores = fg_scores.masked_fill(~valid_sel.unsqueeze(3), float("-inf"))
        o_fg = torch.einsum("bhits,bhisd->bhitd", fg_scores.softmax(dim=-1), v_sel)

        # Local branch: attention within each block independently.
        local_scores = torch.einsum("bhitd,bhisd->bhits", q_b, k_b) * scale
        local_scores = local_scores.masked_fill(~valid_b.unsqueeze(3), float("-inf"))
        o_local = torch.einsum("bhits,bhisd->bhitd", local_scores.softmax(dim=-1), v_b)

        # Gated combination of the three branches (Eq. 2).
        gates = torch.sigmoid(self.gate(x_p))
        outs = torch.stack([o_cg, o_fg, o_local], dim=-1)
        outs = outs.reshape(batch, heads, n_padded, dh, 3).permute(0, 2, 1, 3, 4)
        combined = (outs * gates.view(batch, n_padded, 1, 1, 3)).sum(dim=-1)
        combined = combined.reshape(batch, n_padded, heads * dh)
        out = self.out_proj(combined)
        if pad > 0:
            out = out[:, :n_tokens]
        return out
