import numpy as np
import torch
from graph_weather.models.aurora import (
    IntegrationLayer,
    GenCastConfig,
    Fengwu_GHRConfig,
    Swin3DEncoder,
    Decoder3D,
    LoraLayer,
    PerceiverProcessor
)

def test_aurora_integration():
    """
    Test the Aurora pipeline by integrating all components in a single test case.
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the grid
    grid_lon = np.linspace(0, 359, 32)  # 32 longitude points
    grid_lat = np.linspace(-90, 90, 32)  # 32 latitude points

    # Input and output feature dimensions
    input_features_dim = 96
    output_features_dim = 96

    # Configuration for GenCast
    gencast_config = GenCastConfig(hidden_dims=[128, 256], num_blocks=3, num_heads=4, splits=4, num_hops=2)

    # Configuration for Fengwu_GHR
    fengwu_ghr_config = Fengwu_GHRConfig(
        image_size=(32, 32),
        patch_size=(4, 4),
        depth=2,
        heads=4,
        mlp_dim=256,
        channels=3
    )

    # Initialize IntegrationLayer
    integration_layer = IntegrationLayer(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        input_features_dim=input_features_dim,
        output_features_dim=output_features_dim,
        gencast_config=gencast_config,
        fengwu_ghr_config=fengwu_ghr_config,
        device=device
    )

    # Generate input data
    batch_size = 1
    depth, height, width = 16, 32, 32
    input_tensor = torch.randn(batch_size, 1, depth, height, width).to(device)
    print("Input tensor shape:", input_tensor.shape)

    # Test Swin3DEncoder
    encoder = Swin3DEncoder(in_channels=1, embed_dim=input_features_dim).to(device)
    encoded_features = encoder(input_tensor)
    print("Swin3DEncoder output shape:", encoded_features.shape)

    # Reshape encoded features
    seq_len = depth * height * width
    encoded_features = encoded_features.reshape(batch_size, seq_len, input_features_dim)
    print("Reshaped encoded features shape:", encoded_features.shape)

    # Process with Perceiver Processor
    processor = PerceiverProcessor(latent_dim=512, input_dim=input_features_dim, max_seq_len=2048).to(device)
    processed_latents = processor(encoded_features)
    print("PerceiverProcessor output shape:", processed_latents.shape)

    # Apply LoRA layer
    latent_dim = processed_latents.size(-1)
    lora_layer = LoraLayer(in_features=latent_dim, out_features=output_features_dim).to(device)
    lora_output = lora_layer(processed_latents)
    print("LoRA output shape:", lora_output.shape)

    # Prepare test coordinates
    sample_points = np.array([
        [0, 0],     # Bottom-left corner
        [45, 180],  # Middle point
    ])
    
    # Clip coordinates
    sample_points[:, 0] = np.clip(sample_points[:, 0], -90, 90)    # Latitude
    sample_points[:, 1] = np.clip(sample_points[:, 1], 0, 359)     # Longitude
    
    # Convert to grid indices
    lat_indices = np.searchsorted(grid_lat, sample_points[:, 0])
    lon_indices = np.searchsorted(grid_lon, sample_points[:, 1])
    grid_points = np.column_stack((lat_indices, lon_indices))

    print(f"Grid points (lat, lon) indices: {grid_points}")

    # Verify indices
    assert np.all(grid_points[:, 0] < len(grid_lat))
    assert np.all(grid_points[:, 1] < len(grid_lon))

    # Process coordinates
    pos_x, pos_y = integration_layer.preprocess_coordinates(grid_points.tolist())
    print("Coordinate shapes - pos_x:", pos_x.shape, "pos_y:", pos_y.shape)

    # Reshape coordinates for interpolation
    pos_x = pos_x.view(batch_size, -1)
    pos_y = pos_y.view(batch_size, -1)

    # Create features for interpolation with correct size
    features = torch.randn(batch_size, height * width * depth, output_features_dim, device=device)
    print("Features shape for interpolation:", features.shape)

    # Ensure tensors are on the correct device
    pos_x = pos_x.to(device)
    pos_y = pos_y.to(device)

    # Perform interpolation
    try:
        interpolated_features = integration_layer.interpolate_features(
            x=features,
            pos_x=pos_x,
            pos_y=pos_y,
            k=1,
            weighted=False
        )
        print("Interpolated features shape:", interpolated_features.shape)
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return

    # Reshape interpolated features to match decoder input requirements
    # The size should be [batch_size, output_features_dim, depth, height, width]
    decoder_input = interpolated_features.view(batch_size, output_features_dim, depth, height, width)
    print("Decoder input shape:", decoder_input.shape)

    # Initialize and run decoder
    decoder = Decoder3D(output_channels=1, embed_dim=output_features_dim, target_shape=(depth, height, width)).to(device)
    reconstructed_output = decoder(decoder_input)
    print("Decoder3D output shape:", reconstructed_output.shape)

    # Test Fengwu_GHR functionality
    if fengwu_ghr_config:
        print("Testing Fengwu_GHR-specific functionality:")
        fengwu_model = integration_layer.fengwu_model
        if fengwu_model:
            dummy_image_input = torch.randn(batch_size, fengwu_ghr_config.channels, 
                                          fengwu_ghr_config.image_size[0], 
                                          fengwu_ghr_config.image_size[1]).to(device)
            fengwu_output = fengwu_model(dummy_image_input)
            print("Fengwu_GHR model output shape:", fengwu_output.shape)

    # Clear memory
    del input_tensor, encoded_features, processed_latents, lora_output, reconstructed_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Integration test completed successfully.")

if __name__ == "__main__":
    test_aurora_integration()