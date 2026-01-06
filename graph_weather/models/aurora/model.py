"""Core implementation of Aurora model for unstructured point data.

This module contains the core Aurora model components for processing unstructured
point data in earth system science applications.
"""

from typing import Optional

import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    """Point Encoder for processing unstructured point data.
    
    This encoder handles unstructured point data by combining input features
    with coordinate information.
    """
    
    def __init__(self, input_features: int, embed_dim: int, max_seq_len: int = 1024):
        """Initialize the PointEncoder.
        
        Args:
            input_features (int): Number of input features per point.
            embed_dim (int): Embedding dimension.
            max_seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.input_dim = input_features + 2  # Account for lat/lon coordinates
        self.max_seq_len = max_seq_len

        # Remove positional embeddings as they break point ordering invariance

        # Enhanced coordinate embedding
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Feature embedding
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PointEncoder.
        
        Args:
            points: Point coordinates tensor
            features: Point features tensor
        
        Returns:
            torch.Tensor: Encoded tensor
        """
        num_points = points.shape[1]
        if num_points > self.max_seq_len:
            points = points[:, : self.max_seq_len, :]
            features = features[:, : self.max_seq_len, :]

        # Normalize coordinates to [-1, 1] range
        normalized_points = torch.stack(
            [points[..., 0] / 180.0, points[..., 1] / 90.0],
            dim=-1,  # longitude  # latitude
        )

        # Separately encode coordinates and features
        coord_embedding = self.coord_encoder(normalized_points)
        feature_embedding = self.feature_encoder(features)

        # Combine embeddings through addition (order-invariant operation)
        x = coord_embedding + feature_embedding

        # Final normalization
        x = self.norm(x)

        return x


class PointDecoder(nn.Module):
    """Point Decoder for reconstructing point features from latent representations.

    This decoder takes the processed latent embeddings and reconstructs the output features
    for each point in the point cloud.
    """

    def __init__(self, embed_dim: int, output_features: int):
        """Initialize the PointDecoder.
        
        Args:
            embed_dim: Embedding dimension
            output_features: Number of output features
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PointDecoder.
        
        Args:
            x: (batch_size, num_points, embed_dim) tensor
        
        Returns:
            (batch_size, num_points, output_features) tensor
        """
        return self.decoder(x)


class PointCloudProcessor(nn.Module):
    """Processes point cloud data using self-attention layers.

    This processor applies multiple layers of self-attention to the encoded point cloud
    features to capture spatial and temporal relationships.
    """

    def __init__(self, embed_dim: int, num_layers: int = 4):
        """Initialize the PointCloudProcessor.
        
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of layers
        """
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PointCloudProcessor.
        
        Args:
            x: (batch_size, num_points, embed_dim) tensor
        
        Returns:
            (batch_size, num_points, embed_dim) tensor after processing
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for processing point cloud features.

    This layer applies multi-head self-attention followed by a feed-forward network
    with residual connections and layer normalization.
    """
    
    def __init__(self, embed_dim: int):
        """Initialize the SelfAttentionLayer.
        
        Args:
            embed_dim: Embedding dimension for attention
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SelfAttentionLayer.
        
        Args:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor after attention and normalization
        """
        # First attention block with residual
        x_t = x.transpose(0, 1)
        attended, _ = self.attention(x_t, x_t, x_t)
        attended = attended.transpose(0, 1)
        x = self.norm1(x + attended)

        # FFN block with residual
        x = self.norm2(x + self.ffn(x))
        return x


class EarthSystemLoss(nn.Module):
    """Combined loss function for earth system science predictions.

    This loss function combines MSE loss, spatial correlation loss, and physical
    consistency loss to guide the model toward physically meaningful predictions.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """Initialize the EarthSystemLoss.
        
        Args:
            alpha: Weight for MSE loss
            beta: Weight for spatial correlation loss
            gamma: Weight for physical consistency loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def spatial_correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial correlation loss.
        
        Args:
            pred: Predicted values
            target: Target values
            points: Point coordinates
        
        Returns:
            torch.Tensor: Spatial correlation loss value
        """
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, 2)

        # Compute pairwise distances
        dists = torch.cdist(points_flat, points_flat)
        dists = dists.view(batch_size, num_points, num_points)

        # Create mask for nearby points (5 degrees threshold)
        nearby_mask = (dists < 5.0).float().unsqueeze(-1)

        # Compute differences
        pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)

        # Calculate loss with proper broadcasting
        correlation_loss = torch.mean(nearby_mask * (pred_diff - target_diff).pow(2))

        return correlation_loss

    def physical_loss(self, pred: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Calculate physical consistency loss - ensures predictions follow basic physical laws"""
        # Ensure non-negative values for physical quantities (e.g., temperature in Kelvin)
        min_value_loss = torch.nn.functional.relu(-pred).mean()

        # Ensure reasonable maximum values (e.g., max temperature)
        max_value_loss = torch.nn.functional.relu(pred - 500).mean()  # Assuming max value of 500

        # Add latitude-based consistency (e.g., colder at poles)
        latitude = points[..., 1]  # Second coordinate is latitude
        abs_latitude = torch.abs(latitude)
        latitude_consistency = torch.mean(
            torch.nn.functional.relu(pred[..., 0] - (1.0 - abs_latitude / 90.0) * pred.mean())
        )

        # Combine physical constraints
        physical_loss = min_value_loss + max_value_loss + 0.1 * latitude_consistency
        return physical_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, points: torch.Tensor) -> dict:
        """Forward pass of the EarthSystemLoss.
        
        Args:
            pred: Predicted values
            target: Target values
            points: Point coordinates
        
        Returns:
            Dictionary containing different loss components
        """
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        spatial_loss = self.spatial_correlation_loss(pred, target, points)
        physical_loss = self.physical_loss(pred, points)

        # Combine losses with the specified weights
        total_loss = self.alpha * mse_loss + self.beta * spatial_loss + self.gamma * physical_loss

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "spatial_correlation_loss": spatial_loss,
            "physical_loss": physical_loss,
        }


class AuroraModel(nn.Module):
    """Aurora model for processing unstructured point cloud data in earth system science.

    This model combines point encoding, processing with attention mechanisms, and
    decoding to produce predictions for earth system variables at unstructured points.
    """
    
    def __init__(
        self,
        input_features: int,
        output_features: int,
        latent_dim: int = 256,
        num_layers: int = 4,
        max_points: int = 10000,
        max_seq_len: int = 1024,
        use_checkpointing: bool = False,
    ):
        """Initialize the Aurora model.
        
        Args:
            input_features: Number of input features per point
            output_features: Number of output features per point
            latent_dim: Dimension of the latent space
            num_layers: Number of processing layers
            max_points: Maximum number of points allowed
            max_seq_len: Maximum sequence length
            use_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()

        self.max_points = max_points
        self.max_seq_len = max_seq_len
        self.input_features = input_features
        self.output_features = output_features

        # Model components
        self.encoder = PointEncoder(input_features, latent_dim, max_seq_len)
        self.processor = PointCloudProcessor(latent_dim, num_layers)
        self.decoder = PointDecoder(latent_dim, output_features)

        # Add gradient checkpointing
        self.use_checkpointing = use_checkpointing

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, points: torch.Tensor, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the Aurora model.
        
        Args:
            points: Point coordinates tensor
            features: Point feature tensor
            mask: Optional mask for points
        
        Returns:
            torch.Tensor: Model output predictions
        """
        if points.shape[1] > self.max_points:
            raise ValueError(
                f"Number of points ({points.shape[1]}) exceeds maximum ({self.max_points})"
            )

        # Handle mask properly
        if mask is not None:
            mask = mask.float().unsqueeze(-1)
            points = points * mask
            features = features * mask

        # Forward pass with gradient checkpointing
        x = self.encoder(points, features)

        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.processor, x)
        else:
            x = self.processor(x)

        output = self.decoder(x)

        # Apply mask to output if provided
        if mask is not None:
            output = output * mask

        return output
