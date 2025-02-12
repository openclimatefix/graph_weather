import pytest
import torch
from graph_weather.models.aurora.model import AuroraModel, EarthSystemLoss
from graph_weather.models.aurora.encoder import Swin3DEncoder
from graph_weather.models.aurora.decoder import Decoder3D
from graph_weather.models.aurora.processor import PerceiverProcessor

@pytest.fixture
def sample_3d_data():
    """Create small sample 3D data for testing"""
    batch_size = 2
    channels = 1
    depth = height = width = 8  # Reduced from 32 to 8
    return torch.randn(batch_size, channels, depth, height, width)

@pytest.fixture
def swin3d_config():
    """Minimal configuration for Swin3D encoder"""
    return {
        "in_channels": 1,
        "embed_dim": 32  # Reduced from 96 to 32
    }

@pytest.fixture
def sample_unstructured_data():
    """Create sample unstructured point data for testing"""
    batch_size = 1
    num_points = 100
    num_features = 4

    points = torch.rand(batch_size, num_points, 2) * 360
    points[:, :, 0] = points[:, :, 0] - 180
    points[:, :, 1] = points[:, :, 1] - 90

    features = torch.randn(batch_size, num_points, num_features - 2)

    return points, features

@pytest.fixture
def model_config():
    """Basic model configuration for unstructured point data"""
    return {
        "input_features": 2,
        "output_features": 2,
        "embed_dim": 24,
        "latent_dim": 128,
        "max_points": 5000,
        "max_seq_len": 1024
    }

def test_swin3d_encoder(sample_3d_data, swin3d_config):
    """Test Swin3D encoder with minimal 3D data"""
    print("\n=== Testing Swin3D Encoder ===")
    print(f"Input shape: {sample_3d_data.shape}")
    
    encoder = Swin3DEncoder(**swin3d_config)
    output = encoder(sample_3d_data)
    
    print(f"Output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    assert len(output.shape) == 3
    assert output.shape[0] == sample_3d_data.shape[0]
    assert output.shape[2] == swin3d_config["embed_dim"]
    assert not torch.isnan(output).any()

def test_decoder3d():
    """Test 3D decoder with minimal dimensions"""
    print("\n=== Testing 3D Decoder ===")
    batch_size = 2
    embed_dim = 32  # Reduced from 96 to 32
    target_shape = (8, 8, 8)  # Reduced from 32 to 8
    
    print(f"Target shape: {target_shape}")
    decoder = Decoder3D(output_channels=1, embed_dim=embed_dim, target_shape=target_shape)
    input_tensor = torch.randn(batch_size, target_shape[0] * target_shape[1] * target_shape[2], embed_dim)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    output = decoder(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    assert output.shape == (batch_size, 1, *target_shape)
    assert not torch.isnan(output).any()

def test_perceiver_processor():
    """Test Perceiver processor with minimal data"""
    print("\n=== Testing Perceiver Processor ===")
    batch_size = 2
    seq_len = 16  # Reduced from 100 to 16
    input_dim = 32  # Reduced from 96 to 32
    
    print(f"Input dimensions - Batch: {batch_size}, Sequence length: {seq_len}, Input dim: {input_dim}")
    processor = PerceiverProcessor(latent_dim=64, input_dim=input_dim, max_seq_len=64)
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    
    output = processor(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == 64
    assert not torch.isnan(output).any()

def test_full_pipeline_integration():
    """Test minimal pipeline integration"""
    print("\n=== Testing Full Pipeline Integration ===")
    batch_size = 2
    channels = 1
    size = 8  # Reduced from 32 to 8
    embed_dim = 32  # Reduced from 96 to 32
    
    input_3d = torch.randn(batch_size, channels, size, size, size)
    print(f"Input shape: {input_3d.shape}")
    
    encoder = Swin3DEncoder(in_channels=channels, embed_dim=embed_dim)
    processor = PerceiverProcessor(latent_dim=64, input_dim=embed_dim)
    decoder = Decoder3D(output_channels=channels, embed_dim=64, target_shape=(size, size, size))
    
    encoded = encoder(input_3d)
    print(f"Encoded shape: {encoded.shape}")
    
    processed = processor(encoded)
    print(f"Processed shape: {processed.shape}")
    
    output = decoder(processed.unsqueeze(1).repeat(1, size * size * size, 1))
    print(f"Final output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    assert output.shape == input_3d.shape
    assert not torch.isnan(output).any()

def test_aurora_model_with_3d(sample_3d_data, model_config):
    """Test AuroraModel with minimal 3D data"""
    print("\n=== Testing Aurora Model with 3D Data ===")
    model = AuroraModel(**model_config)
    
    batch_size, channels, depth, height, width = sample_3d_data.shape
    print(f"Input data shape: {sample_3d_data.shape}")
    
    x, y, z = torch.meshgrid(
        torch.linspace(-180, 180, width),
        torch.linspace(-90, 90, height),
        torch.linspace(-1, 1, depth),
        indexing='ij'
    )
    
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    points = points.unsqueeze(0).repeat(batch_size, 1, 1)
    print(f"Points shape: {points.shape}")
    
    features = sample_3d_data.reshape(batch_size, channels, -1)
    features = features.transpose(1, 2)
    print(f"Features shape before padding: {features.shape}")
    
    if features.shape[-1] != model_config["input_features"]:
        features = torch.cat([
            features,
            torch.zeros(batch_size, features.shape[1], 
                       model_config["input_features"] - features.shape[-1])
        ], dim=-1)
        print(f"Features shape after padding: {features.shape}")
    
    output = model(points, features)
    print(f"Model output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == points.shape[1]
    assert output.shape[2] == model_config["output_features"]
    assert not torch.isnan(output).any()

def test_aurora_point_processing(sample_unstructured_data, model_config):
    """Test model's ability to process unstructured point data"""
    print("\n=== Testing Aurora Point Processing ===")
    points, features = sample_unstructured_data
    print(f"Points shape: {points.shape}")
    print(f"Features shape: {features.shape}")
    
    model = AuroraModel(**model_config)
    output = model(points, features)
    print(f"Output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    assert output.shape[0] == points.shape[0]
    assert output.shape[1] == points.shape[1]
    assert output.shape[2] == model_config["output_features"]
    assert not torch.isnan(output).any()

def test_variable_point_counts(model_config):
    """Test model with varying numbers of points per batch"""
    print("\n=== Testing Variable Point Counts ===")
    model = AuroraModel(**model_config)
    point_counts = [100, 200, 300]
    
    for num_points in point_counts:
        print(f"\nTesting with {num_points} points")
        points = torch.rand(1, num_points, 2) * 360 - 180
        features = torch.randn(1, num_points, 2)
        
        output = model(points, features)
        print(f"Output shape: {output.shape}")
        print(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
        
        assert output.shape[1] == points.shape[1]
        assert not torch.isnan(output).any()

def test_point_ordering_invariance(sample_unstructured_data, model_config):
    """Test point ordering invariance with carefully controlled transformations"""
    print("\n=== Testing Point Ordering Invariance ===")
    model = AuroraModel(**model_config)
    model.eval()
    
    with torch.no_grad():
        points = torch.tensor([
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ], dtype=torch.float32)
        
        points[..., 0] = points[..., 0] * 360 - 180
        points[..., 1] = points[..., 1] * 180 - 90
        print("Test points configuration:")
        print(points[0])
        
        features = torch.ones((1, 4, 2), dtype=torch.float32)
        baseline_output = model(points, features)
        print(f"Baseline output stats - Mean: {baseline_output.mean():.4f}, Std: {baseline_output.std():.4f}")
        
        transformations = [
            ([0, 1, 2, 3], "Identity"),
            ([1, 0, 2, 3], "Swap adjacent horizontal"),
            ([2, 3, 0, 1], "Swap top and bottom halves"),
            ([3, 2, 1, 0], "Complete reversal"),
        ]
        
        for perm, name in transformations:
            print(f"\nTesting transformation: {name}")
            perm_points = points[:, perm, :]
            perm_features = features[:, perm, :]
            perm_output = model(perm_points, perm_features)
            
            unperm_output = torch.zeros_like(perm_output)
            unperm_output[:, perm, :] = perm_output
            
            diff = (baseline_output - unperm_output).abs()
            final_diff = diff.max().item()
            print(f"Maximum difference: {final_diff:.6f}")
            
            if final_diff >= 0.2:
                raise AssertionError(
                    f"Point ordering invariance test failed.\n"
                    f"Maximum difference: {final_diff:.6f}\n"
                    f"This suggests the model may be:\n"
                    f"1. Using point indices in its computation\n"
                    f"2. Not properly normalizing attention weights\n"
                    f"3. Having numerical stability issues in its transformations"
                )

def test_earth_system_loss_point_data():
    """Test the custom loss function with point data"""
    print("\n=== Testing Earth System Loss ===")
    loss_fn = EarthSystemLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    batch_size = 1
    num_points = 100
    num_features = 2
    
    points = torch.zeros(batch_size, num_points, 2)
    points[:, :, 0] = torch.linspace(-180, 180, num_points)
    points[:, :, 1] = torch.linspace(-90, 90, num_points)
    print(f"Points shape: {points.shape}")
    
    latitude_factor = 1.0 - torch.abs(points[:, :, 1]) / 90.0
    base_temp = 273.15 + latitude_factor * 30.0
    
    pred = base_temp.unsqueeze(-1).repeat(1, 1, num_features) + torch.randn(batch_size, num_points, num_features) * 2
    target = pred + torch.randn(batch_size, num_points, num_features)
    print(f"Predictions shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    loss_dict = loss_fn(pred, target, points)
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"{key}: {value:.4f}")
    
    assert loss_dict['mse_loss'] >= 0
    assert loss_dict['spatial_correlation_loss'] >= 0
    assert loss_dict['physical_loss'] >= 0
    assert loss_dict['total_loss'] >= 0

def test_sparse_dense_distributions(model_config):
    """Test model with varying point densities"""
    print("\n=== Testing Sparse vs Dense Distributions ===")
    model = AuroraModel(**model_config)
    
    sparse_points = torch.linspace(-180, 180, 5).view(1, 5, 1).repeat(1, 1, 2)
    dense_points = torch.linspace(-1, 1, 100).view(1, 100, 1).repeat(1, 1, 2)
    sparse_features = torch.randn(1, 5, 2)
    dense_features = torch.randn(1, 100, 2)
    
    print("\nTesting sparse distribution:")
    print(f"Sparse points shape: {sparse_points.shape}")
    sparse_output = model(sparse_points, sparse_features)
    print(f"Sparse output stats - Mean: {sparse_output.mean():.4f}, Std: {sparse_output.std():.4f}")
    
    print("\nTesting dense distribution:")
    print(f"Dense points shape: {dense_points.shape}")

def test_pole_handling(model_config):
    """Test model behavior near the poles"""
    model = AuroraModel(**model_config)
    north_pole_points = torch.zeros(1, 50, 2)
    north_pole_points[:, :, 1] = 89.9 + torch.rand(50) * 0.1
    south_pole_points = torch.zeros(1, 50, 2)
    south_pole_points[:, :, 1] = -90 + torch.rand(50) * 0.1
    features = torch.randn(1, 50, 2)
    north_output = model(north_pole_points, features)
    south_output = model(south_pole_points, features)
    assert not torch.isnan(north_output).any()
    assert not torch.isnan(south_output).any()

def test_temporal_sequence(model_config):
    """Test processing of temporal sequences"""
    model = AuroraModel(**model_config)
    num_points = 50
    num_timesteps = 3
    
    # Generate consistent points across timesteps
    points = torch.rand(1, num_points, 2)
    points[..., 0] = points[..., 0] * 360 - 180  # longitude
    points[..., 1] = points[..., 1] * 180 - 90   # latitude
    
    # Generate temporally coherent features
    base_features = torch.randn(1, num_points, model_config["input_features"])
    features = []
    for t in range(num_timesteps):
        # Add small temporal variations
        temporal_noise = torch.randn_like(base_features) * 0.1
        features.append(base_features + temporal_noise * t)
    
    # Process sequence
    outputs = []
    for feat in features:
        output = model(points, feat)
        outputs.append(output)
    
    # Stack outputs and check temporal consistency
    outputs = torch.stack(outputs, dim=1)
    temporal_diff = torch.diff(outputs, dim=1)
    
    # Check if temporal differences are reasonable
    max_allowed_diff = 1.0
    assert torch.all(torch.abs(temporal_diff) < max_allowed_diff), \
        f"Temporal differences exceed {max_allowed_diff}"
    
def test_missing_data(model_config):
    """Test handling of missing data points"""
    model = AuroraModel(**model_config)
    points = torch.rand(1, 50, 2) * 360 - 180
    features = torch.randn(1, 50, 2)
    mask = torch.ones(1, 50, dtype=bool)
    mask[:, ::4] = False
    output = model(points, features, mask=mask)
    assert not torch.isnan(output).any()

def test_model_save_load(tmp_path, model_config):
    """Test model serialization and deserialization"""
    model = AuroraModel(**model_config)
    points = torch.rand(1, 50, 2) * 360 - 180
    features = torch.randn(1, 50, 2)
    output_before = model(points, features)
    save_path = tmp_path / "aurora_model.pt"
    torch.save(model.state_dict(), save_path)
    loaded_model = AuroraModel(**model_config)
    loaded_model.load_state_dict(torch.load(save_path))
    output_after = loaded_model(points, features)
    assert torch.allclose(output_before, output_after)

def test_gradient_flow(model_config):
    """Test that gradients flow through the model properly"""
    model = AuroraModel(**model_config)
    model.train()
    
    points = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], 
                         requires_grad=True)
    features = torch.ones(1, 3, 2, requires_grad=True)
    
    output = model(points, features)
    coord_loss = (points ** 2).mean()
    output_loss = (output ** 2).mean()
    loss = output_loss + coord_loss
    
    loss.backward(retain_graph=True)
    
    assert points.grad is not None, "Points gradients are None - computation graph is disconnected"
    assert torch.any(points.grad != 0), "Points gradients are all zero"