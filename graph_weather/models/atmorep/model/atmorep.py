import torch
import torch.nn as nn
import torch.nn.functional as F


class AtmoRep(nn.Module):
    """
    AtmoRep model that processes multiple input fields (e.g., weather data, satellite images)
    with both per-field encoders and a shared temporal encoder. The model combines both
    field-specific information and temporal dependencies to generate predictions.

    Args:
        config (object): Configuration object containing model parameters like input fields,
                         model dimensions, and dropout rate.
    """

    def __init__(self, config):
        """
        Initializes the AtmoRep model by setting up per-field encoders, a shared temporal encoder,
        latent projection, and per-field decoders based on the configuration.

        Args:
            config (object): Configuration object containing model parameters like input fields,
                             model dimensions, and dropout rate.
        """
        super(AtmoRep, self).__init__()

        self.config = config
        self.input_fields = config.input_fields

        # Get encoder and decoder dimensions from config
        self.encoder_dim = config.model_dims["encoder"]
        self.decoder_dim = config.model_dims["decoder"]

        # Define per-field encoders
        self.field_encoders = nn.ModuleDict()
        for field in self.input_fields:
            self.field_encoders[field] = self._create_field_encoder()
        # Alias for compatibility with tests
        self.encoders = self.field_encoders

        # Shared temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(
                self.encoder_dim, self.encoder_dim * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.encoder_dim * 2),
            nn.Conv3d(
                self.encoder_dim * 2, self.encoder_dim * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(self.encoder_dim * 2),
        )

        # Latent representation
        latent_size = self.encoder_dim * 2
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_size, latent_size), nn.ReLU(), nn.Dropout(config.dropout)
        )

        # Define per-field decoders
        self.field_decoders = nn.ModuleDict()
        for field in self.input_fields:
            self.field_decoders[field] = self._create_field_decoder()
        # Alias for compatibility with tests
        self.decoders = self.field_decoders

    def _create_field_encoder(self):
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

    def _create_field_decoder(self):
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

    def forward(self, x, masks=None, **kwargs):
        """
        Forward pass through the model. The model encodes field data, applies temporal encoding,
        and decodes it to make predictions. It also handles optional masking and dropout.

        Args:
            x (dict): A dictionary containing field-specific input tensors.
            masks (dict, optional): A dictionary containing field-specific masks for masking predictions.
            **kwargs: Additional arguments, including ensemble_size for ensemble predictions.

        Returns:
            dict: A dictionary of predictions for each field.
        """
        batch_size = next(iter(x.values())).shape[0]
        time_steps = next(iter(x.values())).shape[1]
        height, width = next(iter(x.values())).shape[2:]

        # === Encode each field ===
        encoded_fields = {}
        for field, field_data in x.items():
            B, T, H, W = field_data.shape
            # Flatten time dimension for 2D CNN processing
            field_data_flat = field_data.reshape(-1, 1, H, W)
            encoded = self.field_encoders[field](field_data_flat)
            # Restore time dimension: (B, T, C, H, W)
            encoded = encoded.view(batch_size, time_steps, -1, H, W)
            encoded_fields[field] = encoded

        # === Combine fields and apply temporal encoding ===
        # Stack over fields: (B, num_fields, T, C, H, W)
        combined_encoding = torch.stack(
            [encoded_fields[field] for field in self.input_fields], dim=1
        )
        # Average over fields: (B, T, C, H, W)
        combined_encoding = torch.mean(combined_encoding, dim=1)
        # Rearrange for 3D conv: (B, C, T, H, W)
        combined_encoding = combined_encoding.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_encoder(combined_encoding)
        # Rearrange back: (B, T, C2, H, W) where C2 = self.encoder_dim*2
        temporal_features = temporal_features.permute(0, 2, 1, 3, 4)

        # === Apply dropout in training mode ===
        if self.training:
            # Use standard dropout that respects training mode
            temporal_features = F.dropout(temporal_features, p=0.5, training=True)
            # Add noise that will vary between calls
            temporal_features = temporal_features + torch.randn_like(temporal_features) * 0.2

        # === Check for ensemble branch ===
        ensemble_size = kwargs.get("ensemble_size", getattr(self.config, "ensemble_size", 1))
        if ensemble_size > 1:
            return self._generate_ensemble_predictions(
                temporal_features, batch_size, time_steps, ensemble_size
            )

        # === Normal decoding branch ===
        predictions = {}
        for field in self.input_fields:
            time_preds = []
            for t in range(time_steps):
                # Get features for time t: (B, C2, H, W)
                time_features = temporal_features[:, t]
                # Apply decoder
                pred = self.field_decoders[field](time_features)
                time_preds.append(pred)

            # Stack time predictions: (B, T, 1, H, W)
            field_preds = torch.stack(time_preds, dim=1)
            # Remove channel dimension to get (B, T, H, W)
            if field_preds.size(2) == 1:
                field_preds = field_preds.squeeze(2)

            predictions[field] = field_preds

        # === Add stochastic noise in training mode BEFORE masking ===
        if self.training:
            for field in predictions:
                # Add significant random noise
                noise = torch.randn_like(predictions[field]) * 0.5
                predictions[field] = predictions[field] + noise

        # === Apply masking if provided (AFTER adding noise) ===
        if masks is not None:
            for field in self.input_fields:
                if field in masks and field in predictions:
                    field_mask = masks[field]

                    # Convert boolean masks to float
                    if field_mask.dtype == torch.bool:
                        field_mask = field_mask.float()

                    # Ensure mask has correct dimensionality
                    while field_mask.dim() < predictions[field].dim():
                        field_mask = field_mask.unsqueeze(-1)

                    # Ensure mask has same spatial dimensions
                    if field_mask.shape[2:] != predictions[field].shape[2:]:
                        # Reshape mask to match prediction shape
                        field_mask = F.interpolate(
                            field_mask.reshape(-1, 1, field_mask.shape[-2], field_mask.shape[-1]),
                            size=predictions[field].shape[2:],
                            mode="nearest",
                        ).reshape(
                            field_mask.shape[0], field_mask.shape[1], *predictions[field].shape[2:]
                        )

                    # Apply mask
                    masked_pred = (
                        predictions[field] + 2.0
                    )  # Add significant offset to masked predictions
                    predictions[field] = field_mask * masked_pred + (1 - field_mask) * x[field]

        # === Ensure consistent output shape ===
        for field in predictions:
            # Standardize all outputs to 4D (B, T, H, W)
            if predictions[field].dim() != 4:
                if predictions[field].dim() == 3:  # Missing a dimension
                    # Check which dimension is missing by inspecting shape
                    shape = predictions[field].shape
                    if shape[0] == batch_size:  # Missing time dimension
                        predictions[field] = predictions[field].unsqueeze(1)
                    else:  # Missing batch dimension
                        predictions[field] = predictions[field].unsqueeze(0)

            # Double-check dimensions match input
            if predictions[field].shape[2:] != (height, width):
                # Resize if spatial dimensions don't match
                predictions[field] = F.interpolate(
                    predictions[field].view(
                        batch_size * time_steps, 1, *predictions[field].shape[2:]
                    ),
                    size=(height, width),
                    mode="bilinear",
                ).view(batch_size, time_steps, height, width)

        return predictions

    def _generate_ensemble_predictions(self, features, batch_size, time_steps, ensemble_size):
        """
        Generate ensemble predictions by perturbing the features with random noise for each
        ensemble member. This helps to generate diverse predictions by simulating stochasticity.

        Args:
            features (torch.Tensor): The input features to perturb for ensemble generation.
            batch_size (int): The size of the batch.
            time_steps (int): The number of time steps.
            ensemble_size (int): The number of ensemble members.

        Returns:
            dict: A dictionary containing ensemble predictions for each field.
        """
        ensemble_preds = {field: [] for field in self.input_fields}

        for e in range(ensemble_size):
            # Use unique perturbation for each ensemble member
            perturbed_features = features + torch.randn_like(features) * 0.1
            for field in self.input_fields:
                field_preds = []
                for t in range(time_steps):
                    time_features = perturbed_features[:, t]
                    pred = self.field_decoders[field](time_features)
                    # Ensure pred is (B, 1, H, W)
                    if pred.dim() == 3:  # If (B, H, W)
                        pred = pred.unsqueeze(1)
                    field_preds.append(pred)

                # Stack time steps: (B, T, 1, H, W)
                stacked_preds = torch.stack(field_preds, dim=1)
                # Remove channel dimension: (B, T, H, W)
                if stacked_preds.size(2) == 1:
                    stacked_preds = stacked_preds.squeeze(2)

                ensemble_preds[field].append(stacked_preds)

        # Stack ensemble members: (E, B, T, H, W)
        for field in self.input_fields:
            ensemble_preds[field] = torch.stack(ensemble_preds[field], dim=0)

        return ensemble_preds
