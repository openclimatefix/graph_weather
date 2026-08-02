# MOSAIC

Unofficial implementation of components from
[(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](https://arxiv.org/abs/2604.16429)
(Zhdanov et al.).

This subpackage provides the two parts requested in issue #217: block-sparse
attention and processing at the native grid resolution. It is not a
reproduction of the full forecaster.

## Components

| Module | Paper reference | Purpose |
| --- | --- | --- |
| `BlockSparseAttention` | Eq. 2, 8-11, Section 4.2 | Three-branch attention: compression, block-level top-n selection, and within-block local attention, combined by learned gating |
| `RotaryEmbedding2D` | Section 4.3 | 2D axial rotary embedding over (lat, lon) |
| `CrossAttentionInterpolator` | Eq. 6-7, Section 4.1 | Moves features between point sets using relative-position queries |
| `HealpixCoarsen` / `HealpixRefine` | Eq. 12-13, Section 4.3 | Learnable pooling and unpooling of four sibling pixels |
| `MosaicTransformerBlock` / `MosaicProcessor` | Eq. 14, Section 4.3 | Pre-norm block and a multi-scale processor whose first stage runs at native resolution |

## Token ordering

Block-sparse attention assumes contiguous index blocks are spatial
neighbourhoods. On a HEALPix grid this holds in NESTED ordering, where pixel
`p` subdivides into children `4p ... 4p+3` (Section 3.2). For irregular point
sets, `healpix.nested_order` returns a permutation that provides the same
property; any other locality-preserving ordering works as well, so the
attention module never requires `healpy`.

## Scope

Implemented: the attention mechanism, grid transfer, hierarchical pooling and
a processor that can be used standalone.

Not implemented: the full forecaster and its 82 channel input/output, CRPS
training, ensemble noise injection, the Triton kernel used for large
sequences on GPU, and pretrained weights.

## Provenance

Written from the paper text and equations only. The reference implementation
is published without a license, so no code from it was consulted.
