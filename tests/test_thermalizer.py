import torch

from graph_weather.models.layers.thermalizer import ThermalizerLayer


def test_thermalizer_forward_shape():
    """Test thermalizer with explicit dimensions"""
    batch_size = 2
    height, width = 12, 12
    nodes = height * width  # 144 nodes
    features = 256

    # Create input in the expected format: (batch * nodes, features)
    total_samples = batch_size * nodes
    x = torch.randn(total_samples, features)

    print(f"Input shape: {x.shape}")
    print(f"Grid dimensions: {height}x{width} = {nodes} nodes")
    print(f"Batch size: {batch_size}")

    # Initialize thermalizer
    layer = ThermalizerLayer(input_dim=features)
    t = torch.randint(0, 1000, (1,)).item()

    print(f"Timestep: {t}")

    # Test with explicit dimensions
    out = layer(x, t, height=height, width=width, batch=batch_size)

    # Verify output shape matches input
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # Verify output is valid
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert torch.isfinite(out).all(), "Output contains infinite values"

    print(f"Output shape: {out.shape}")
    print("✓ Explicit dimensions test passed!")


def test_thermalizer_auto_inference():
    """Test thermalizer with automatic dimension inference"""
    batch_size = 2
    height, width = 12, 12
    nodes = height * width
    features = 256

    # Create input in the expected format: (batch * nodes, features)
    total_samples = batch_size * nodes
    x = torch.randn(total_samples, features)

    print(f"\nInput shape: {x.shape}")
    print("Testing auto-inference (no explicit dimensions)")

    # Initialize thermalizer
    layer = ThermalizerLayer(input_dim=features)
    t = torch.randint(0, 1000, (1,)).item()

    # Test with auto-inference (should assume batch=1 and infer grid)
    out = layer(x, t)

    # Verify output shape matches input
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert torch.isfinite(out).all(), "Output contains infinite values"

    print(f"Output shape: {out.shape}")
    print("✓ Auto-inference test passed!")


def test_thermalizer_different_sizes():
    """Test thermalizer with various grid sizes"""
    test_cases = [
        (1, 4, 2, 2),  # Small: 1 batch, 4 nodes (2x2)
        (1, 9, 3, 3),  # Medium: 1 batch, 9 nodes (3x3)
        (1, 16, 4, 4),  # Medium: 1 batch, 16 nodes (4x4)
        (2, 25, 5, 5),  # Larger: 2 batches, 25 nodes (5x5)
        (3, 64, 8, 8),  # Large: 3 batches, 64 nodes (8x8)
    ]

    features = 256
    layer = ThermalizerLayer(input_dim=features)

    for batch_size, nodes, height, width in test_cases:
        print(
            f"\n=== Testing {batch_size} batch(es), {height}x{width} grid " f"({nodes} nodes) ==="
        )

        total_samples = batch_size * nodes
        x = torch.randn(total_samples, features)
        t = torch.randint(0, 1000, (1,)).item()

        # Test with explicit dimensions
        out_explicit = layer(x, t, height=height, width=width, batch=batch_size)
        assert out_explicit.shape == x.shape
        assert not torch.isnan(out_explicit).any()

        # Test with auto-inference (only for single batch to avoid ambiguity)
        if batch_size == 1:
            out_auto = layer(x, t)
            assert out_auto.shape == x.shape
            assert not torch.isnan(out_auto).any()
            print(f"✓ Both explicit and auto-inference passed for {height}x{width}")
        else:
            print(f"✓ Explicit dimensions passed for {height}x{width}")


def test_grid_reconstruction():
    """Test that we can properly reconstruct spatial structure"""
    batch_size = 1
    height, width = 6, 8  # Non-square grid
    nodes = height * width
    features = 256

    print(f"\n=== Testing grid reconstruction {height}x{width} ===")

    # Create input with spatial pattern
    x_grid = torch.randn(batch_size, features, height, width)

    # Convert to flat format
    x_flat = x_grid.permute(0, 2, 3, 1).reshape(batch_size * nodes, features)

    print(f"Original grid shape: {x_grid.shape}")
    print(f"Flattened shape: {x_flat.shape}")

    # Apply thermalizer
    layer = ThermalizerLayer(input_dim=features)
    t = 100  # Fixed timestep

    out_flat = layer(x_flat, t, height=height, width=width, batch=batch_size)

    # Reconstruct grid
    out_grid = out_flat.reshape(batch_size, height, width, features).permute(0, 3, 1, 2)

    print(f"Output flat shape: {out_flat.shape}")
    print(f"Reconstructed grid shape: {out_grid.shape}")

    # Verify shapes
    assert out_flat.shape == x_flat.shape
    assert out_grid.shape == x_grid.shape
    assert not torch.isnan(out_grid).any()

    print("✓ Grid reconstruction test passed!")


if __name__ == "__main__":
    test_thermalizer_forward_shape()
    test_thermalizer_auto_inference()
    test_thermalizer_different_sizes()
    test_grid_reconstruction()
    print("\nAll tests passed!")
