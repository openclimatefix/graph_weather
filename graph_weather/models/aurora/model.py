"""
aurora/model.py - Core implementation of Aurora model for unstructured point data
"""

from typing import Optional

import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    """
    Embed unstructured points and their features into a common latent space.

    Coordinates and features are embedded by two separate multi-layer perceptrons and are
    combined by addition, an order-invariant operation. No positional embeddings are used,
    so the result does not depend on the ordering of the points.
    """

    def __init__(self, input_features: int, embed_dim: int, max_seq_len: int = 1024):
        """
        Initialize the point encoder.

        Args:
            input_features (int): Number of feature channels attached to each point.
            embed_dim (int): Size of the embedding produced for each point.
            max_seq_len (int): Maximum number of points kept per sample. Longer inputs are
                truncated in ``forward``.
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
        """
        Encode point coordinates and point features into a single embedding per point.

        Args:
            points (torch.Tensor): Coordinates of shape (batch_size, num_points, 2), given as
                (longitude, latitude) in degrees.
            features (torch.Tensor): Features of shape (batch_size, num_points, input_features).

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, num_points, embed_dim). Inputs
            longer than ``max_seq_len`` are truncated to their first ``max_seq_len`` points.
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
    """Decodes latent representations back to point features."""

    def __init__(self, embed_dim: int, output_features: int):
        """
        Initialize the point decoder.

        Args:
            embed_dim (int): Size of the latent embedding of each point.
            output_features (int): Number of feature channels produced for each point.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent point embeddings into output features.

        Args:
            x: (batch_size, num_points, embed_dim) tensor

        Returns:
            (batch_size, num_points, output_features) tensor
        """
        return self.decoder(x)


class PointCloudProcessor(nn.Module):
    """Processes point cloud data using self-attention layers."""

    def __init__(self, embed_dim: int, num_layers: int = 4):
        """
        Initialize the point cloud processor.

        Args:
            embed_dim (int): Size of the latent embedding of each point.
            num_layers (int): Number of stacked self-attention layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the stack of self-attention layers to the point embeddings.

        Args:
            x: (batch_size, num_points, embed_dim) tensor

        Returns:
            (batch_size, num_points, embed_dim) tensor after processing
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SelfAttentionLayer(nn.Module):
    """
    Transformer block combining multi-head self-attention and a feed-forward network.

    Each of the two sub-blocks is wrapped in a residual connection followed by layer
    normalization.
    """

    def __init__(self, embed_dim: int):
        """
        Initialize the self-attention layer.

        Args:
            embed_dim (int): Size of the per-point embedding, also used as the attention
                model dimension. The layer uses 8 attention heads.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention and the feed-forward network with residual connections.

        The input is transposed internally because ``nn.MultiheadAttention`` is created with
        its default sequence-first layout.

        Args:
            x (torch.Tensor): Point embeddings of shape (batch_size, num_points, embed_dim).

        Returns:
            torch.Tensor: Updated embeddings of shape (batch_size, num_points, embed_dim).
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
    """
    Composite loss for point-wise Earth system predictions.

    It sums a mean squared error term, a spatial correlation term that compares the
    differences between nearby points, and a physical consistency term, weighted by
    ``alpha``, ``beta`` and ``gamma`` respectively.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Initialize the loss weights.

        Args:
            alpha (float): Weight of the mean squared error term.
            beta (float): Weight of the spatial correlation term.
            gamma (float): Weight of the physical consistency term.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def spatial_correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize disagreement between predicted and target differences of nearby points.

        Distances are Euclidean in the raw (longitude, latitude) coordinates and pairs of
        points closer than 5 degrees are treated as neighboring.

        Args:
            pred (torch.Tensor): Predictions of shape (batch_size, num_points, features).
            target (torch.Tensor): Targets of shape (batch_size, num_points, features).
            points (torch.Tensor): Coordinates of shape (batch_size, num_points, 2).

        Returns:
            torch.Tensor: Scalar tensor with the mean squared difference taken over the
            neighboring point pairs.
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
        """
        Compute the weighted total loss together with each of its terms.

        Args:
            pred (torch.Tensor): Predictions of shape (batch_size, num_points, features).
            target (torch.Tensor): Targets of shape (batch_size, num_points, features).
            points (torch.Tensor): Coordinates of shape (batch_size, num_points, 2).

        Returns:
            dict: Mapping with the keys ``total_loss``, ``mse_loss``,
            ``spatial_correlation_loss`` and ``physical_loss``, each holding a scalar tensor.
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
    """
    Encoder-processor-decoder model for predicting features on unstructured points.

    Points and their features are embedded by ``PointEncoder``, refined by
    ``PointCloudProcessor`` and mapped back to feature space by ``PointDecoder``.
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
        """
        Initialize the model and its encoder, processor and decoder.

        Args:
            input_features (int): Number of feature channels attached to each input point.
            output_features (int): Number of feature channels predicted for each point.
            latent_dim (int): Size of the latent embedding used throughout the model.
            num_layers (int): Number of self-attention layers in the processor.
            max_points (int): Maximum number of points accepted by ``forward``.
            max_seq_len (int): Maximum number of points kept by the encoder.
            use_checkpointing (bool): Whether to run the processor under gradient
                checkpointing during training to save memory.
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
        """
        Predict output features for a batch of points.

        Args:
            points (torch.Tensor): Coordinates of shape (batch_size, num_points, 2), given as
                (longitude, latitude) in degrees.
            features (torch.Tensor): Features of shape (batch_size, num_points, input_features).
            mask (Optional[torch.Tensor]): Mask of shape (batch_size, num_points). Masked
                positions are zeroed in the inputs and in the output.

        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_points, output_features).

        Raises:
            ValueError: If the number of points exceeds ``max_points``.
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
