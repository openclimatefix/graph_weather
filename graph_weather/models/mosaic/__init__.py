"""MOSAIC block-sparse attention and native-grid processing.

Modules from "(Sparse) Attention to the Details: Preserving Spectral
Fidelity in ML-based Weather Forecasting Models" (arXiv:2604.16429).

Everything here is pure PyTorch and operates on point sets of shape
(batch, n_tokens, dim), so the components can be reused by the graph-based
models in this repository. See README.md in this directory for scope.
"""

from .block_sparse_attention import BlockSparseAttention, RotaryEmbedding2D
from .coarsen import HealpixCoarsen, HealpixRefine
from .interpolate import CrossAttentionInterpolator, knn_indices
from .layers import MosaicProcessor, MosaicTransformerBlock, SwiGLU

__all__ = [
    "BlockSparseAttention",
    "CrossAttentionInterpolator",
    "HealpixCoarsen",
    "HealpixRefine",
    "MosaicProcessor",
    "MosaicTransformerBlock",
    "RotaryEmbedding2D",
    "SwiGLU",
    "knn_indices",
]
