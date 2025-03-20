from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AtmoRep(nn.Module):
    """
    AtmoRep model that processes multiple input fields (e.g., weather data, satellite images)
    with both per-field encoders and a shared temporal encoder. The model combines both
    field-specific information and temporal dependencies to generate predictions.

    Args:
        config: Configuration object containing model parameters like:
            - input_fields (List[str]): Fields the model expects.
            - model_dims (Dict[str, int]): e.g., {"encoder": 64, "decoder": 64}
            - dropout (float): Dropout rate for latent projection.
            - ensemble_size (int): Number of ensemble members for multi-sample predictions.
    """

    # If you prefer a single config object:
    def __init__(self, config: Any) -> None:
        """
        Initializes the AtmoRep model by setting up per-field encoders, a shared temporal encoder,
        latent projection, and per-field decoders based on the configuration.

        Args:
            config: A configuration object or dictionary with required attributes:
                    input_fields, model_dims, dropout, ensemble_size, etc.
        """
        super().__init__()

        self.config = config
        self.input_fields = config.input_fields

        # Get encoder and decoder dimensions from config
        self.encoder_dim = config.model_dims["encoder"]
        self.decoder_dim = config.model_dims["decoder"]
        self.dropout_rate = getattr(config, "dropout", 0.1)
        self.ensemble_size = getattr(config, "ensemble_size", 1)

        # Define per-field encoders
        self.field_encoders = nn.ModuleDict()
        for field in self.input_fields:
            self.field_encoders[field] = self._create_field_encoder()
        # Alias for compatibility with older tests
        self.encoders = self.field_encoders

        # Shared temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(self.encoder_dim, self.encoder_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(self.encoder_dim * 2),
            nn.Conv3d(self.encoder_dim * 2, self.encoder_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(self.encoder_dim * 2),
        )

        # Latent projection
        latent_size = self.encoder_dim * 2
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # Define per-field decoders
        self.field_decoders = nn.ModuleDict()
        for field in self.input_fields:
            self.field_decoders[field] = self._create_field_decoder()
        # Alias for compatibility with older tests
        self.decoders = self.field_decoders

    def _create_field_encoder(self) -> nn.Sequential:
        """
        Creates a CNN encoder for a specific field. This encoder applies convolutional layers
        followed by ReLU activations and Batch Normalization.

        Returns:
            nn.Sequential: A CNN encoder for the field.
        """
        return nn.Sequential(
            nn.Conv2d(1, self.encoder_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.encoder_dim // 2),
            nn.Conv2d(self.encoder_dim // 2, self.encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.encoder_dim),
        )

    def _create_field_decoder(self) -> nn.Sequential:
        """
        Creates a CNN decoder for a specific field. This decoder applies convolutional layers
        followed by ReLU activations and Batch Normalization, and reconstructs the output.

        Returns:
            nn.Sequential: A CNN decoder for the field.
        """
        return nn.Sequential(
            nn.Conv2d(self.encoder_dim * 2, self.decoder_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.decoder_dim),
            nn.Conv2d(self.decoder_dim, self.decoder_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.decoder_dim // 2),
            nn.Conv2d(self.decoder_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model. The model encodes field data, applies temporal encoding,
        and decodes it to make predictions. It also handles optional masking and dropout.

        Args:
            x: A dictionary containing field-specific input tensors, each of shape (B, T, H, W).
            masks: A dictionary containing field-specific masks for masking predictions.
            **kwargs: Additional arguments, including 'ensemble_size' for ensemble predictions.

        Returns:
            A dictionary of predictions for each field, each shaped (B, T, H, W).
        """
        # === Determine batch/time/spatial sizes from first field ===
        sample_field = next(iter(x))
        batch_size, time_steps, height, width = x[sample_field].shape

        # === Encode each field ===
        encoded_fields = {}
        for field, field_data in x.items():
            # Flatten time dimension for 2D CNN: (B*T, 1, H, W)
            field_data_flat = rearrange(field_data, "b t h w -> (b t) 1 h w")

            # Encode
            encoded = self.field_encoders[field](field_data_flat)

            # Unflatten back to (B, T, C, H, W)
            encoded = rearrange(encoded, "(b t) c h w -> b t c h w", b=batch_size, t=time_steps)
            encoded_fields[field] = encoded

        # === Combine fields and apply temporal encoding ===
        # Stack over fields => shape: (B, num_fields, T, C, H, W)
        combined_encoding = torch.stack([encoded_fields[f] for f in self.input_fields], dim=1)
        # Average over the field dimension => (B, T, C, H, W)
        combined_encoding = combined_encoding.mean(dim=1)
        # Rearrange for 3D conv => (B, C, T, H, W)
        combined_encoding = rearrange(combined_encoding, "b t c h w -> b c t h w")

        # Pass through 3D conv
        temporal_features = self.temporal_encoder(combined_encoding)

        # Rearrange back => (B, T, C2, H, W) where C2 = encoder_dim*2
        temporal_features = rearrange(temporal_features, "b c t h w -> b t c h w")

        # Optionally apply dropout/noise in training mode
        if self.training:
            temporal_features = F.dropout(temporal_features, p=0.5, training=True)
            temporal_features = temporal_features + torch.randn_like(temporal_features) * 0.2

        # === Check for ensemble predictions ===
        ensemble_size = kwargs.get("ensemble_size", self.ensemble_size)
        if ensemble_size > 1:
            return self._generate_ensemble_predictions(
                temporal_features, batch_size, time_steps, ensemble_size
            )

        # === Normal decoding branch ===
        predictions: Dict[str, torch.Tensor] = {}
        for field in self.input_fields:
            time_preds = []
            for t in range(time_steps):
                # Extract features for time t => (B, C2, H, W)
                time_features = temporal_features[:, t]
                # Decode => (B, 1, H, W)
                pred = self.field_decoders[field](time_features)
                time_preds.append(pred)

            # Stack => (B, T, 1, H, W), then squeeze channel => (B, T, H, W)
            field_preds = torch.stack(time_preds, dim=1).squeeze(2)
            predictions[field] = field_preds

        # === Add stochastic noise in training mode BEFORE masking ===
        if self.training:
            for field in predictions:
                noise = torch.randn_like(predictions[field]) * 0.5
                predictions[field] += noise

        # === Apply masking if provided (AFTER noise) ===
        if masks is not None:
            for field, mask in masks.items():
                if field not in predictions:
                    continue

                # Convert bool -> float
                if mask.dtype == torch.bool:
                    mask = mask.float()

                # Expand mask dims if needed
                while mask.dim() < predictions[field].dim():
                    mask = mask.unsqueeze(-1)

                # Reshape mask if spatial dims differ
                if mask.shape[2:] != predictions[field].shape[2:]:
                    mask = F.interpolate(
                        rearrange(mask, "b t h w -> (b t) 1 h w"),
                        size=predictions[field].shape[2:],
                        mode="nearest",
                    )
                    mask = rearrange(mask, "(b t) 1 h w -> b t h w", b=batch_size, t=time_steps)

                # Apply mask: add offset of +2.0 to masked predictions
                masked_pred = predictions[field] + 2.0
                predictions[field] = mask * masked_pred + (1 - mask) * x[field]

        # === Final check: ensure shape is (B, T, H, W) ===
        for field in predictions:
            if predictions[field].dim() == 3:
                # If missing the time dimension => insert
                if predictions[field].shape[0] == batch_size:
                    predictions[field] = predictions[field].unsqueeze(1)

            # Resize if final spatial dims mismatch
            if predictions[field].shape[2:] != (height, width):
                predictions[field] = F.interpolate(
                    rearrange(predictions[field], "b t h w -> (b t) 1 h w"),
                    size=(height, width),
                    mode="bilinear",
                )
                predictions[field] = rearrange(
                    predictions[field], "(b t) 1 h w -> b t h w", b=batch_size, t=time_steps
                )

        return predictions

    def _generate_ensemble_predictions(
        self, features: torch.Tensor, batch_size: int, time_steps: int, ensemble_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Generate ensemble predictions by perturbing features with random noise for each
        ensemble member. This helps produce diverse predictions by simulating stochasticity.

        Args:
            features: Tensor of shape (B, T, C2, H, W).
            batch_size: Batch size.
            time_steps: Number of time steps.
            ensemble_size: Number of ensemble members.

        Returns:
            A dictionary of ensemble predictions for each field, each shaped (E, B, T, H, W).
        """
        ensemble_preds = {field: [] for field in self.input_fields}

        for _ in range(ensemble_size):
            # Add small noise for each ensemble member
            perturbed_features = features + torch.randn_like(features) * 0.1
            member_preds: Dict[str, torch.Tensor] = {}

            # Decode each field
            for field in self.input_fields:
                time_preds = []
                for t in range(time_steps):
                    time_features = perturbed_features[:, t]
                    pred = self.field_decoders[field](time_features)
                    if pred.dim() == 3:  # shape (B, H, W)
                        pred = pred.unsqueeze(1)  # shape (B, 1, H, W)
                    time_preds.append(pred)

                # (B, T, 1, H, W) => (B, T, H, W)
                stacked_preds = torch.stack(time_preds, dim=1).squeeze(2)
                member_preds[field] = stacked_preds

            # Accumulate this ensemble member's predictions
            for field in self.input_fields:
                ensemble_preds[field].append(member_preds[field])

        # Stack ensemble members => (E, B, T, H, W)
        for field in ensemble_preds:
            ensemble_preds[field] = torch.stack(ensemble_preds[field], dim=0)

        return ensemble_preds
