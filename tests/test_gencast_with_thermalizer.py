import torch
from graph_weather.models.forecast import GraphWeatherForecaster


def test_gencast_thermal_integration():
    # Test with a larger grid to properly test UNet capabilities
    # Create a 3x3 grid (9 nodes)
    lat_lons = [(i // 3, i % 3) for i in range(9)]  # 3x3 grid

    print(f"Number of lat_lons: {len(lat_lons)}")
    print(f"Grid should be: 3x3")

    model = GraphWeatherForecaster(
        lat_lons,
        use_thermalizer=True,
        feature_dim=3,
        aux_dim=0,
        node_dim=256,  # Match the thermalizer input_dim
        num_blocks=1,  # Reduce complexity for testing
    )

    print(f"Model grid_shape: {model.grid_shape}")

    features = torch.randn(1, len(lat_lons), 3)

    print(f"Input features shape: {features.shape}")

    try:
        # Run GenCast to get one-step prediction
        pred = model(features, t=torch.randint(0, 1000, (1,)).item())

        # Check output shape - should match input
        print(f"Input shape: {features.shape}")
        print(f"Output shape: {pred.shape}")

        # The output might be in grid format if constraints are applied
        if pred.dim() == 4:  # [B, C, H, W] format
            assert pred.shape[0] == features.shape[0]  # Same batch size
            assert (
                pred.shape[2] * pred.shape[3] == features.shape[1]
            )  # Same number of spatial points
            print("Output is in grid format [B, C, H, W]")
        else:  # [B, N, C] format
            assert pred.shape == features.shape
            print("Output is in node format [B, N, C]")

        # Check for valid outputs
        assert not torch.isnan(pred).any(), "Output contains NaN values"
        assert torch.isfinite(pred).all(), "Output contains infinite values"

        print("Test passed!")

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()
        raise


def test_thermalizer_standalone():
    """Test the thermalizer layer independently with different grid sizes"""
    from graph_weather.models.layers.thermalizer import ThermalizerLayer

    print("Testing thermalizer with different grid sizes...")

    # Test with small grid (2x2)
    print("\n=== Testing 2x2 grid ===")
    batch, nodes, features = 1, 4, 256
    x = torch.randn(batch * nodes, features)

    layer = ThermalizerLayer(input_dim=features)
    t = torch.randint(0, 1000, (1,)).item()

    print(f"Input shape: {x.shape}")
    print(f"Timestep: {t}")

    try:
        out = layer(x, t)  # Should auto-infer dimensions and use simple network
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        print("✓ 2x2 grid test passed!")
    except Exception as e:
        print(f"✗ 2x2 grid test failed: {e}")
        raise

    # Test with medium grid (3x3)
    print("\n=== Testing 3x3 grid ===")
    batch, nodes, features = 1, 9, 256
    x = torch.randn(batch * nodes, features)

    try:
        out = layer(x, t)  # Should auto-infer dimensions and use simple network
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        print("✓ 3x3 grid test passed!")
    except Exception as e:
        print(f"✗ 3x3 grid test failed: {e}")
        raise

    # Test with larger grid (8x8) - should use full UNet
    print("\n=== Testing 8x8 grid ===")
    batch, nodes, features = 1, 64, 256
    x = torch.randn(batch * nodes, features)

    try:
        out = layer(x, t)  # Should auto-infer dimensions and use full UNet
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        print("✓ 8x8 grid test passed!")
    except Exception as e:
        print(f"✗ 8x8 grid test failed: {e}")
        raise


def test_small_grid_integration():
    """Test the original small grid case"""
    print("\n=== Testing original 2x2 grid integration ===")
    lat_lons = [(i // 2, i % 2) for i in range(4)]  # 2x2 grid

    model = GraphWeatherForecaster(
        lat_lons, use_thermalizer=True, feature_dim=3, aux_dim=0, node_dim=256, num_blocks=1
    )

    features = torch.randn(1, len(lat_lons), 3)

    try:
        pred = model(features, t=50)  # Fixed timestep for reproducibility
        print(f"Input shape: {features.shape}")
        print(f"Output shape: {pred.shape}")

        assert not torch.isnan(pred).any()
        assert torch.isfinite(pred).all()
        print("✓ Small grid integration test passed!")

    except Exception as e:
        print(f"✗ Small grid integration test failed: {e}")
        raise


if __name__ == "__main__":
    print("Testing thermalizer standalone...")
    test_thermalizer_standalone()
    print("\nTesting small grid integration...")
    test_small_grid_integration()
    print("\nTesting larger grid integration...")
    test_gencast_thermal_integration()

    from graph_weather.models.layers.thermalizer import ThermalizerLayer

    batch, nodes, features = 1, 4, 256
    x = torch.randn(batch * nodes, features)
    layer = ThermalizerLayer(input_dim=features)
    print(f"Thermalizer input shape: {x.shape}")
    t = torch.randint(0, 1000, (1,)).item()
    print(f"Timestep: {t}")

    try:
        out = layer(x, t)  # Should auto-infer dimensions
        print(f"Thermalizer output shape: {out.shape}")
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        print("Thermalizer standalone test passed!")
    except Exception as e:
        print(f"Thermalizer error: {e}")
        import traceback

        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Testing thermalizer standalone...")
    test_thermalizer_standalone()
    print("\nTesting full model integration...")
    test_gencast_thermal_integration()
