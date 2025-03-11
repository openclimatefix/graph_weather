import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import AtmoRepConfig
from .attention import CrossAttention, SpatioTemporalAttention
from .decoder import Decoder
from .field_transformer import FieldVisionTransformer


# MultiFormer: Multiple Vision Transformers with cross-attention
class MultiFormer(nn.Module):
    """
    MultiFormer model that processes multiple fields of data using Vision Transformers
    and applies cross-attention mechanisms to combine features from different fields.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: AtmoRepConfig):
        super().__init__()
        self.config = config

        # Create a ViT for each input field
        self.field_transformers = nn.ModuleDict(
            {field: FieldVisionTransformer(config, field) for field in config.input_fields}
        )

        # Cross-attention mechanisms between fields
        self.cross_attentions = nn.ModuleDict(
            {
                f"{field1}_to_{field2}": CrossAttention(config)
                for field1 in config.input_fields
                for field2 in config.input_fields
                if field1 != field2
            }
        )

        # Decoder for each field
        self.decoders = nn.ModuleDict({field: Decoder(config) for field in config.input_fields})

        # Ensemble prediction heads
        self.ensemble_heads = nn.ModuleDict(
            {
                field: nn.ModuleList(
                    [nn.Linear(config.hidden_dim, 1) for _ in range(config.num_ensemble_members)]
                )
                for field in config.input_fields
            }
        )

    def forward(self, x_dict, mask_dict=None):
        """
        Forward pass of the MultiFormer model.

        Args:
            x_dict (dict): A dictionary where each key corresponds to a field and
                           the value is the input tensor of shape [B, T, H, W].
            mask_dict (dict, optional): A dictionary where each key corresponds to a field
                                         and the value is the mask tensor of shape [B, T, N]. Default is None.

        Returns:
            dict: A dictionary of ensemble predictions for each field.
        """
        # Encode each field
        encoded_features = {}
        multi_scale_features = {}

        for field_name, field_data in x_dict.items():
            field_mask = mask_dict.get(field_name) if mask_dict else None
            encoded, features = self.field_transformers[field_name](field_data, field_mask)
            encoded_features[field_name] = encoded
            multi_scale_features[field_name] = features

        # Apply cross-attention between fields
        cross_attended_features = {field: feat.clone() for field, feat in encoded_features.items()}

        for field1 in self.config.input_fields:
            for field2 in self.config.input_fields:
                if field1 != field2:
                    cross_key = f"{field2}_to_{field1}"
                    cross_attn = self.cross_attentions[cross_key]
                    cross_attended_features[field1] += cross_attn(
                        encoded_features[field1], encoded_features[field2]
                    )

        # Decode each field
        decoded_features = {}
        for field_name, field_encoded in cross_attended_features.items():
            decoded = self.decoders[field_name](field_encoded, multi_scale_features[field_name])
            decoded_features[field_name] = decoded

        # Apply ensemble prediction heads
        predictions = {}
        for field_name, field_decoded in decoded_features.items():
            field_preds = []
            for head in self.ensemble_heads[field_name]:
                # Reshape to the original spatial dimensions
                B, T, N, D = field_decoded.shape
                H = W = int(math.sqrt(N) * self.config.patch_size)
                pred = head(field_decoded)
                pred = rearrange(
                    pred, "b t (h w) c -> b t (h c) w", h=int(math.sqrt(N)), w=int(math.sqrt(N))
                )
                pred = F.interpolate(pred, size=(H, W), mode="bilinear")
                pred = rearrange(pred, "b t c w -> b t w c")
                field_preds.append(pred)

            # Stack ensemble predictions
            predictions[field_name] = torch.stack(field_preds, dim=0)  # [E, B, T, H, W]

        return predictions


# Enhanced MultiFormer with spatiotemporal attention
class EnhancedMultiFormer(MultiFormer):
    """
    EnhancedMultiFormer extends the MultiFormer model by adding spatiotemporal attention
    after the regular transformer blocks to better capture the temporal and spatial dependencies
    in the input data.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: AtmoRepConfig):
        super().__init__(config)

        # Add spatiotemporal attention after regular transformer blocks
        self.spatiotemporal_attns = nn.ModuleDict(
            {field: SpatioTemporalAttention(config) for field in config.input_fields}
        )

    def forward(self, x_dict, mask_dict=None):
        """
        Forward pass of the EnhancedMultiFormer model, which includes spatiotemporal attention.

        Args:
            x_dict (dict): A dictionary where each key corresponds to a field and
                           the value is the input tensor of shape [B, T, H, W].
            mask_dict (dict, optional): A dictionary where each key corresponds to a field
                                         and the value is the mask tensor of shape [B, T, N]. Default is None.

        Returns:
            dict: A dictionary of ensemble predictions for each field after applying
                  both cross-attention and spatiotemporal attention.
        """
        # Get basic MultiFormer output
        encoded_features, multi_scale_features = {}, {}

        for field_name, field_data in x_dict.items():
            field_mask = mask_dict.get(field_name) if mask_dict else None
            encoded, features = self.field_transformers[field_name](field_data, field_mask)
            encoded_features[field_name] = encoded
            multi_scale_features[field_name] = features

        # Apply cross-attention between fields
        cross_attended_features = {field: feat.clone() for field, feat in encoded_features.items()}

        for field1 in self.config.input_fields:
            for field2 in self.config.input_fields:
                if field1 != field2:
                    cross_key = f"{field2}_to_{field1}"
                    cross_attn = self.cross_attentions[cross_key]
                    cross_attended_features[field1] += cross_attn(
                        encoded_features[field1], encoded_features[field2]
                    )

        # Apply spatiotemporal attention
        spatiotemporal_features = {}
        for field_name, field_feature in cross_attended_features.items():
            spatiotemporal_features[field_name] = self.spatiotemporal_attns[field_name](
                field_feature
            )

        # Decode each field
        decoded_features = {}
        for field_name, field_encoded in spatiotemporal_features.items():
            decoded = self.decoders[field_name](field_encoded, multi_scale_features[field_name])
            decoded_features[field_name] = decoded

        # Apply ensemble prediction heads
        predictions = {}
        for field_name, field_decoded in decoded_features.items():
            field_preds = []
            for head in self.ensemble_heads[field_name]:
                # Reshape to the original spatial dimensions
                B, T, N, D = field_decoded.shape
                H = W = int(math.sqrt(N) * self.config.patch_size)
                pred = head(field_decoded)
                pred = rearrange(
                    pred, "b t (h w) c -> b t (h c) w", h=int(math.sqrt(N)), w=int(math.sqrt(N))
                )
                pred = F.interpolate(pred, size=(H, W), mode="bilinear")
                pred = rearrange(pred, "b t c w -> b t w c")
                field_preds.append(pred)

            # Stack ensemble predictions
            predictions[field_name] = torch.stack(field_preds, dim=0)  # [E, B, T, H, W]

        return predictions


class DataParallelAtmoRep(nn.Module):
    """
    A wrapper class to distribute the AtmoRep model across multiple GPUs for parallel processing
    during training.

    Args:
        config (AtmoRepConfig): Configuration object containing model parameters.
        num_gpus (int, optional): Number of GPUs to use for data parallel training. Default is None, which
                                   uses all available GPUs.
    """

    def __init__(self, config: AtmoRepConfig, num_gpus=None):
        super().__init__()
        self.config = config
        self.model = EnhancedMultiFormer(config)

        # Determine number of GPUs to use
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        self.num_gpus = min(num_gpus, torch.cuda.device_count())

        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs for data parallel training")
            self.model = nn.DataParallel(self.model)

    def forward(self, x_dict, mask_dict=None):
        """
        Forward pass for the DataParallelAtmoRep model.

        Args:
            x_dict (dict): A dictionary where each key corresponds to a field and
                           the value is the input tensor of shape [B, T, H, W].
            mask_dict (dict, optional): A dictionary where each key corresponds to a field
                                         and the value is the mask tensor of shape [B, T, N]. Default is None.

        Returns:
            dict: A dictionary of ensemble predictions for each field.
        """
        return self.model(x_dict, mask_dict)
