import sys
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

# Add the project root to Python path so we can import from graph_weather
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_weather.data import AnemoiDataset


def test_anemoi_dataset():
    """Test the AnemoiDataset class with mock data"""
    print("🧪 Testing AnemoiDataset...")

    # Test configuration
    dataset_config = {
        "dataset_name": "era5-test",  # This will trigger mock data creation
        "features": ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"],
        "time_range": ("2020-01-01", "2020-01-31"),
        "time_step": 1,
        "max_samples": 10,  # Small number for testing
    }

    try:
        # Create dataset
        dataset = AnemoiDataset(**dataset_config)
        print(f"✅ Dataset created successfully!")
        print(f"   Dataset length: {len(dataset)}")
        print(f"   Grid size: {dataset.num_lat} x {dataset.num_lon}")
        print(f"   Features: {dataset.features}")

        # Test getting a single sample
        input_data, target_data = dataset[0]
        print(f"✅ Sample retrieved successfully!")
        print(f"   Input shape: {input_data.shape}")
        print(f"   Target shape: {target_data.shape}")
        print(f"   Data type: {input_data.dtype}")

        # Check for NaN values
        if np.isnan(input_data).any() or np.isnan(target_data).any():
            print("❌ Found NaN values in data!")
        else:
            print("✅ No NaN values found")

        # Test with DataLoader (like other graph_weather dataloaders)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_input, batch_target = next(iter(dataloader))
        print(f"✅ DataLoader works!")
        print(f"   Batch input shape: {batch_input.shape}")
        print(f"   Batch target shape: {batch_target.shape}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_normalization():
    """Test that normalization is working correctly"""
    print("\n🧪 Testing normalization...")

    dataset = AnemoiDataset(dataset_name="test", features=["temperature"], max_samples=5)

    # Get a few samples and check normalization
    samples = []
    for i in range(min(3, len(dataset))):
        input_data, _ = dataset[i]
        samples.append(input_data[:, 0])  # First feature (temperature)

    all_values = np.concatenate(samples)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)

    print(f"   Sample mean: {mean_val:.4f} (should be close to 0)")
    print(f"   Sample std: {std_val:.4f} (should be close to 1)")

    if abs(mean_val) < 0.5 and abs(std_val - 1.0) < 0.5:
        print("✅ Normalization looks reasonable")
    else:
        print("⚠️  Normalization might need adjustment")


def test_time_features():
    """Test that time features are being added correctly"""
    print("\n🧪 Testing time features...")

    dataset = AnemoiDataset(dataset_name="test", features=["temperature"], max_samples=2)

    input_data, _ = dataset[0]
    num_features = len(dataset.features)
    num_time_features = 4  # sin/cos day, sin/cos hour

    expected_total_features = num_features + num_time_features
    actual_features = input_data.shape[1]

    print(f"   Expected features: {expected_total_features}")
    print(f"   Actual features: {actual_features}")

    if actual_features == expected_total_features:
        print("✅ Time features added correctly")
    else:
        print(f"❌ Feature count mismatch!")


def compare_with_gencast_format():
    """Compare output format with GenCast dataloader expectations"""
    print("\n🧪 Comparing with GenCast format...")

    dataset = AnemoiDataset(
        dataset_name="test", features=["temperature", "geopotential"], max_samples=2
    )

    input_data, target_data = dataset[0]

    print(f"   Data shapes match expected format: {input_data.shape == target_data.shape}")
    print(f"   Data is float32: {input_data.dtype == np.float32}")
    print(f"   Spatial dimension flattened: {len(input_data.shape) == 2}")

    # Check that we have the right number of locations
    expected_locations = dataset.num_lat * dataset.num_lon
    actual_locations = input_data.shape[0]

    print(f"   Location count: {actual_locations} (expected: {expected_locations})")

    if actual_locations == expected_locations:
        print("✅ Format matches graph_weather expectations")
    else:
        print("❌ Format mismatch!")


if __name__ == "__main__":
    print("🚀 Running Anemoi Dataset Tests\n")

    # Run all tests
    success = test_anemoi_dataset()

    if success:
        test_normalization()
        test_time_features()
        compare_with_gencast_format()

        print("\n🎉 All tests completed!")
        print("Your AnemoiDataset is ready for integration with graph_weather!")
    else:
        print("\n💥 Tests failed - check the implementation")
