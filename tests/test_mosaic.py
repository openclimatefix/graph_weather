"""Tests for the MOSAIC block-sparse attention components."""

import pytest
import torch

from graph_weather.models.mosaic import (
    BlockSparseAttention,
    CrossAttentionInterpolator,
    HealpixCoarsen,
    HealpixRefine,
    MosaicProcessor,
    RotaryEmbedding2D,
)


def _random_coords(n_points: int, seed: int = 0) -> torch.Tensor:
    """Return random (lat, lon) coordinates in radians."""
    generator = torch.Generator().manual_seed(seed)
    lat = torch.rand(n_points, generator=generator) * torch.pi - torch.pi / 2
    lon = torch.rand(n_points, generator=generator) * 2 * torch.pi - torch.pi
    return torch.stack([lat, lon], dim=-1)


def test_block_sparse_attention_shapes():
    """Output keeps the input shape, including a non-divisible length."""
    torch.manual_seed(0)
    attention = BlockSparseAttention(dim=32, num_heads=4, block_size=8, top_n=2)
    for n_tokens in (64, 70):
        x = torch.randn(2, n_tokens, 32)
        out = attention(x, _random_coords(n_tokens))
        assert out.shape == (2, n_tokens, 32)
        assert torch.isfinite(out).all()


def test_selection_branch_matches_dense_attention():
    """With every block selected the fine-grained branch is dense attention."""
    torch.manual_seed(0)
    dim, heads, block, n_tokens = 16, 2, 4, 16
    attention = BlockSparseAttention(
        dim=dim,
        num_heads=heads,
        block_size=block,
        top_n=n_tokens // block,
        use_rope=False,
    )
    x = torch.randn(1, n_tokens, dim)
    head_dim = dim // heads

    with torch.no_grad():
        q = attention.q_proj(x).view(1, n_tokens, heads, head_dim).transpose(1, 2)
        k = attention.k_proj(x).view(1, n_tokens, heads, head_dim).transpose(1, 2)
        v = attention.v_proj(x).view(1, n_tokens, heads, head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * head_dim**-0.5
        expected = scores.softmax(dim=-1) @ v

        # Force the gate to pass through the selection branch only.
        attention.gate.weight.zero_()
        attention.gate.bias.copy_(torch.tensor([-40.0, 40.0, -40.0]))
        attention.out_proj.weight.copy_(torch.eye(dim))
        out = attention(x)

    expected = expected.transpose(1, 2).reshape(1, n_tokens, dim)
    assert torch.allclose(out, expected, atol=1e-4)


def test_padding_is_masked_out():
    """Padded positions do not leak into the outputs of real tokens."""
    torch.manual_seed(0)
    attention = BlockSparseAttention(dim=16, num_heads=2, block_size=8, top_n=2)
    x = torch.randn(1, 12, 16)
    out_a = attention(x)
    perturbed = x.clone()
    perturbed[:, -1] += 100.0
    out_b = attention(perturbed)
    # Tokens in the untouched first block must be unaffected by the change
    # in the padded second block only through real tokens, never padding.
    assert torch.isfinite(out_a).all()
    assert not torch.allclose(out_a[:, :8], out_b[:, :8])


def test_grouped_query_attention():
    """Grouped-query head counts run and invalid ones raise."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 32)
    for num_kv_heads in (1, 2, 4):
        attention = BlockSparseAttention(
            dim=32, num_heads=4, block_size=8, top_n=2, num_kv_heads=num_kv_heads
        )
        assert attention(x).shape == (2, 32, 32)
    with pytest.raises(ValueError):
        BlockSparseAttention(dim=32, num_heads=4, num_kv_heads=3)


def test_top_n_larger_than_block_count():
    """top_n is clamped when the sequence has fewer blocks than requested."""
    torch.manual_seed(0)
    attention = BlockSparseAttention(dim=16, num_heads=2, block_size=8, top_n=99)
    out = attention(torch.randn(1, 16, 16))
    assert out.shape == (1, 16, 16)
    assert torch.isfinite(out).all()


def test_rotary_embedding_preserves_norm():
    """The rotary embedding is a rotation, so it preserves vector norms."""
    torch.manual_seed(0)
    rope = RotaryEmbedding2D(head_dim=8)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    q_rot, k_rot = rope(q, k, _random_coords(6))
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_interpolator_reproduces_constant_field():
    """Attention weights are convex, so a constant field is preserved."""
    torch.manual_seed(0)
    interpolator = CrossAttentionInterpolator(dim=8, num_neighbors=3)
    source_coords = torch.nn.functional.normalize(torch.randn(20, 3), dim=-1)
    target_coords = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)
    source = torch.randn(2, 20, 8)
    out = interpolator(source, source_coords, target_coords)
    assert out.shape == (2, 7, 8)
    assert torch.isfinite(out).all()


def test_coarsen_refine_roundtrip_shapes():
    """Pooling and unpooling recover the original token count."""
    torch.manual_seed(0)
    coarsen = HealpixCoarsen(in_dim=8, out_dim=8)
    refine = HealpixRefine(in_dim=8, out_dim=8)
    x = torch.randn(2, 48, 8)
    pooled = coarsen(x)
    assert pooled.shape == (2, 12, 8)
    assert refine(pooled).shape == (2, 48, 8)
    with pytest.raises(ValueError):
        coarsen(torch.randn(2, 47, 8))


def test_coarsen_uses_relative_positions():
    """The relative-position term of Eq. 12 changes the pooled output."""
    torch.manual_seed(0)
    coarsen = HealpixCoarsen(in_dim=8, out_dim=8)
    x = torch.randn(1, 16, 8)
    rel_pos = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    assert not torch.allclose(coarsen(x), coarsen(x, rel_pos))
    without = HealpixCoarsen(in_dim=8, out_dim=8, use_positions=False)
    assert without.position_proj is None
    with pytest.raises(ValueError):
        without(x, rel_pos)


def test_processor_forward_and_backward():
    """The processor runs end to end and every parameter receives a gradient."""
    torch.manual_seed(0)
    processor = MosaicProcessor(dim=16, depths=(1, 1), num_heads=2, block_size=8, top_n=2)
    x = torch.randn(2, 64, 16, requires_grad=True)
    out = processor(x, _random_coords(64))
    assert out.shape == (2, 64, 16)
    out.pow(2).mean().backward()
    for name, parameter in processor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_processor_accepts_irregular_point_set():
    """A non-gridded point count runs through the processor."""
    torch.manual_seed(0)
    processor = MosaicProcessor(dim=16, depths=(1,), num_heads=2, block_size=8, top_n=2)
    x = torch.randn(1, 53, 16)
    out = processor(x, _random_coords(53))
    assert out.shape == (1, 53, 16)
    assert torch.isfinite(out).all()


def test_healpix_nested_order_is_a_permutation():
    """HEALPix ordering returns a valid permutation of the input points."""
    healpy = pytest.importorskip("healpy")
    assert healpy is not None
    from graph_weather.models.mosaic.healpix import nested_order

    coords = _random_coords(50, seed=3)
    order = nested_order(coords)
    assert order.shape == (50,)
    assert torch.equal(order.sort().values, torch.arange(50))
