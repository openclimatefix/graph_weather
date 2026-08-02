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


def test_padding_never_influences_real_tokens():
    """Whatever the padding holds, outputs for real tokens are unchanged."""
    torch.manual_seed(0)
    attention = BlockSparseAttention(dim=16, num_heads=2, block_size=8, top_n=2)
    x = torch.randn(1, 12, 16)
    coords = _random_coords(12)
    baseline = attention(x, coords)

    # The module pads internally with zeros; replace the padding with a
    # large value to prove masked positions cannot leak into the result.
    import graph_weather.models.mosaic.block_sparse_attention as bsa

    original_pad = bsa.F.pad

    def poisoned_pad(tensor, pad, *args, **kwargs):
        padded = original_pad(tensor, pad, *args, **kwargs)
        if pad[-1] > 0:
            padded[:, tensor.shape[1] :] = 1234.5
        return padded

    bsa.F.pad = poisoned_pad
    try:
        poisoned = attention(x, coords)
    finally:
        bsa.F.pad = original_pad

    assert torch.allclose(baseline, poisoned, atol=1e-6)


def test_selection_matches_naive_loop_when_sparse():
    """The gathered blocks match a loop reference for top_n < n_blocks."""
    torch.manual_seed(3)
    batch, heads, n_tokens, dim, block, top_n = 2, 2, 24, 16, 4, 2
    attention = BlockSparseAttention(
        dim=dim, num_heads=heads, block_size=block, top_n=top_n, use_rope=False
    )
    x = torch.randn(batch, n_tokens, dim)
    head_dim = dim // heads
    n_blocks = n_tokens // block

    with torch.no_grad():
        shape = (batch, n_tokens, heads, head_dim)
        q = attention.q_proj(x).view(shape).transpose(1, 2)
        k = attention.k_proj(x).view(shape).transpose(1, 2)
        v = attention.v_proj(x).view(shape).transpose(1, 2)
        q = q.reshape(batch, heads, n_blocks, block, head_dim)
        k = k.reshape(batch, heads, n_blocks, block, head_dim)
        v = v.reshape(batch, heads, n_blocks, block, head_dim)
        scale = head_dim**-0.5
        scores = torch.einsum("bhid,bhjd->bhij", q.mean(3), k.mean(3)) * scale
        selected = scores.mean(dim=1).topk(top_n, dim=-1).indices

        reference = torch.zeros(batch, heads, n_blocks, block, head_dim)
        for b in range(batch):
            for h in range(heads):
                for i in range(n_blocks):
                    keys = torch.cat([k[b, h, j] for j in selected[b, i]], dim=0)
                    values = torch.cat([v[b, h, j] for j in selected[b, i]], dim=0)
                    weights = ((q[b, h, i] @ keys.T) * scale).softmax(-1)
                    reference[b, h, i] = weights @ values

        attention.gate.weight.zero_()
        attention.gate.bias.copy_(torch.tensor([-40.0, 40.0, -40.0]))
        attention.out_proj.weight.copy_(torch.eye(dim))
        out = attention(x)

    reference = reference.reshape(batch, heads, n_tokens, head_dim)
    reference = reference.permute(0, 2, 1, 3).reshape(batch, n_tokens, dim)
    assert torch.allclose(out, reference, atol=1e-4)


def test_training_memory_is_not_quadratic():
    """Doubling the token count must not quadruple the backward memory."""
    import resource

    def peak_delta(n_tokens: int) -> float:
        torch.manual_seed(0)
        attention = BlockSparseAttention(dim=64, num_heads=2, block_size=64, top_n=2)
        x = torch.randn(1, n_tokens, 64, requires_grad=True)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        attention(x).pow(2).mean().backward()
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return max(after - before, 1)

    small = peak_delta(2048)
    large = peak_delta(4096)
    # Quadratic growth would be about 4x; allow generous headroom while
    # still failing if the (n_blocks, n_blocks) expansion is materialised.
    assert large < small * 3


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

    # Bypass the value and output projections so the convex combination is
    # observable directly in the returned features.
    with torch.no_grad():
        interpolator.v_proj.weight.copy_(torch.eye(8))
        interpolator.out_proj.weight.copy_(torch.eye(8))
        interpolator.norm.weight.fill_(1.0)
        constant = torch.full((2, 20, 8), 0.75)
        out = interpolator(constant, source_coords, target_coords)

    assert out.shape == (2, 7, 8)
    normalised = torch.nn.functional.rms_norm(constant, (8,))
    assert torch.allclose(out, normalised[:, :7], atol=1e-5)


def test_interpolator_gradient_is_finite_at_coincident_points():
    """A target sitting exactly on a source point keeps gradients bounded."""
    torch.manual_seed(0)
    interpolator = CrossAttentionInterpolator(dim=8, num_neighbors=2)
    source_coords = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    target_coords = source_coords[:2].clone().requires_grad_(True)
    interpolator(torch.randn(1, 5, 8), source_coords, target_coords).sum().backward()
    assert torch.isfinite(target_coords.grad).all()
    assert target_coords.grad.abs().max() < 1e3


def test_coarsen_refine_roundtrip_shapes():
    """Pooling and unpooling recover the original token count."""
    torch.manual_seed(0)
    coarsen = HealpixCoarsen(in_dim=8, out_dim=8, use_positions=False)
    refine = HealpixRefine(in_dim=8, out_dim=8, use_positions=False)
    x = torch.randn(2, 48, 8)
    pooled = coarsen(x)
    assert pooled.shape == (2, 12, 8)
    assert refine(pooled).shape == (2, 48, 8)
    with pytest.raises(ValueError):
        coarsen(torch.randn(2, 47, 8))


def test_coarsen_position_contract():
    """Positions change the output and the layer refuses mismatched use."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 8)
    rel_pos = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)

    with_positions = HealpixCoarsen(in_dim=8, out_dim=8)
    without = HealpixCoarsen(in_dim=8, out_dim=8, use_positions=False)
    assert without.position_proj is None
    assert not torch.allclose(with_positions(x, rel_pos), without(x))

    # A layer built for positions must be given them, and vice versa, so
    # no projection can silently sit unused.
    with pytest.raises(ValueError):
        with_positions(x)
    with pytest.raises(ValueError):
        without(x, rel_pos)


def test_processor_rejects_bad_token_count():
    """The error names the input size and the required divisor."""
    processor = MosaicProcessor(dim=16, depths=(1, 1, 1), num_heads=2, block_size=8, top_n=2)
    with pytest.raises(ValueError, match="52 must be divisible by 16"):
        processor(torch.randn(1, 52, 16))


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
