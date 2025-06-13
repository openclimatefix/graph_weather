import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import CrossAttention, SpatioTemporalAttention
from .decoder import Decoder
from .field_transformer import FieldVisionTransformer


class MultiFormer(nn.Module):
    """
    MultiFormer processes multiple fields of data using Vision Transformers (ViTs)
    and applies cross-attention to combine features from different fields.

    Args:
        input_fields (List[str]): Names of the input fields (e.g. ["temperature", "humidity"]).
        hidden_dim (int): Dimensionality of ViT patch embeddings and attention features.
        patch_size (int): Spatial patch size used by each FieldVisionTransformer.
        spatial_dims (Tuple[int, int]): The (height, width) of input data for each field.
        time_steps (int): Number of time steps in each field's input.
        num_ensemble_members (int): How many ensemble heads to apply per field.
        cross_attn_num_heads (int): Number of heads for cross-attention layers.
        cross_attn_dropout (float): Dropout probability in cross-attention.
        decoder_num_layers (int): Number of layers in each Decoder.
        decoder_num_heads (int): Number of attention heads in each Decoder's transformer blocks.
        decoder_dropout (float): Dropout probability within the Decoder's transformer blocks.
    """

    def __init__(
        self,
        input_fields: List[str],
        hidden_dim: int,
        patch_size: int,
        spatial_dims: Tuple[int, int],
        time_steps: int,
        num_ensemble_members: int,
        cross_attn_num_heads: int,
        cross_attn_dropout: float,
        decoder_num_layers: int,
        decoder_num_heads: int,
        decoder_dropout: float,
    ) -> None:
        super().__init__()

        self.input_fields = input_fields
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        self.time_steps = time_steps
        self.num_ensemble_members = num_ensemble_members

        # Create a Vision Transformer for each input field
        self.field_transformers = nn.ModuleDict(
            {
                field: FieldVisionTransformer(
                    field_name=field,
                    hidden_dim=hidden_dim,
                    patch_size=patch_size,
                    spatial_dims=spatial_dims,
                    time_steps=time_steps,
                )
                for field in input_fields
            }
        )

        # Create cross-attention layers between fields
        self.cross_attentions = nn.ModuleDict(
            {
                f"{f2}_to_{f1}": CrossAttention(
                    hidden_dim=hidden_dim,
                    num_heads=cross_attn_num_heads,
                    attention_dropout=cross_attn_dropout,
                )
                for f1 in input_fields
                for f2 in input_fields
                if f1 != f2
            }
        )

        # Create a decoder for each field
        self.decoders = nn.ModuleDict(
            {
                field: Decoder(
                    hidden_dim=hidden_dim,
                    num_layers=decoder_num_layers,
                    num_heads=decoder_num_heads,
                    dropout=decoder_dropout,
                )
                for field in input_fields
            }
        )

        # Create ensemble heads (linear layers) for each field
        self.ensemble_heads = nn.ModuleDict(
            {
                field: nn.ModuleList(
                    [nn.Linear(hidden_dim, 1) for _ in range(num_ensemble_members)]
                )
                for field in input_fields
            }
        )

    def forward(
        self, x_dict: Dict[str, torch.Tensor], mask_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MultiFormer model.

        Args:
            x_dict (dict): Keys are field names; values are [B, T, H, W] tensors.
            mask_dict (dict, optional): Keys are field names; values are [B, T, N] masks
                (where N is the number of patches). Default is None.

        Returns:
            dict: Ensemble predictions for each field. Each entry is a tensor shaped [E, B, T, H, W],
                  where E = num_ensemble_members.
        """
        # === 1) Encode each field with its Vision Transformer ===
        encoded_features = {}
        multi_scale_features = {}
        for field_name, field_data in x_dict.items():
            field_mask = mask_dict[field_name] if (mask_dict and field_name in mask_dict) else None
            encoded, features = self.field_transformers[field_name](field_data, field_mask)
            encoded_features[field_name] = encoded
            multi_scale_features[field_name] = features

        # === 2) Apply cross-attention among fields ===
        cross_attended_features = {field: feat.clone() for field, feat in encoded_features.items()}
        for f1 in self.input_fields:
            for f2 in self.input_fields:
                if f1 != f2:
                    cross_key = f"{f2}_to_{f1}"
                    cross_attn_layer = self.cross_attentions[cross_key]
                    cross_attended_features[f1] += cross_attn_layer(
                        encoded_features[f1], encoded_features[f2]
                    )

        # === 3) Decode each field using multi-scale features ===
        decoded_features = {}
        for field_name, feat in cross_attended_features.items():
            decoded = self.decoders[field_name](feat, multi_scale_features[field_name])
            decoded_features[field_name] = decoded

        # === 4) Apply ensemble prediction heads ===
        predictions = {}
        for field_name, field_decoded in decoded_features.items():
            B, T, N, D = field_decoded.shape
            # Convert patch tokens back to 2D images of shape (H, W)
            patch_grid = int(math.sqrt(N))
            H = W = patch_grid * self.patch_size

            field_preds = []
            for head in self.ensemble_heads[field_name]:
                # head: nn.Linear -> [B, T, N, D] -> [B, T, N, 1]
                pred = head(field_decoded)

                # Rearrange => [B, T, (H*C), W], then upsample
                pred = rearrange(pred, "b t (h w) c -> b t (h c) w", h=patch_grid, w=patch_grid)
                pred = F.interpolate(pred, size=(H, W), mode="bilinear")

                # Rearrange => [B, T, W, H] if desired (or keep as channels)
                # We'll treat last dimension as 'C' or 'Channels=1'
                pred = rearrange(pred, "b t c w -> b t w c")
                field_preds.append(pred)

            # Stack across ensemble dimension => [E, B, T, H, W]
            predictions[field_name] = torch.stack(field_preds, dim=0)

        return predictions


class EnhancedMultiFormer(MultiFormer):
    """
    EnhancedMultiFormer extends MultiFormer by adding spatiotemporal attention
    after the regular transformer blocks to better capture temporal/spatial dependencies.

    Args:
        input_fields (List[str]): Names of the input fields (e.g. ["temperature", "humidity"]).
        hidden_dim (int): Dimensionality of ViT patch embeddings and attention features.
        patch_size (int): Spatial patch size used by each FieldVisionTransformer.
        spatial_dims (Tuple[int, int]): The (height, width) of input data for each field.
        time_steps (int): Number of time steps in each field's input.
        num_ensemble_members (int): How many ensemble heads to apply per field.
        cross_attn_num_heads (int): Number of heads for cross-attention layers.
        cross_attn_dropout (float): Dropout probability in cross-attention.
        decoder_num_layers (int): Number of layers in each Decoder.
        decoder_num_heads (int): Number of attention heads in each Decoder's transformer blocks.
        decoder_dropout (float): Dropout probability within the Decoder's transformer blocks.
        spatio_num_heads (int): Number of heads for spatiotemporal attention.
        spatio_attn_dropout (float): Dropout probability in spatiotemporal attention. Default is 0.1.
        spatio_output_dropout (float): Dropout probability after spatiotemporal attention's output. Default is 0.1.
    """

    def __init__(
        self,
        input_fields: List[str],
        hidden_dim: int,
        patch_size: int,
        spatial_dims: Tuple[int, int],
        time_steps: int,
        num_ensemble_members: int,
        cross_attn_num_heads: int,
        cross_attn_dropout: float,
        decoder_num_layers: int,
        decoder_num_heads: int,
        decoder_dropout: float,
        spatio_num_heads: int,
        spatio_attn_dropout: float = 0.1,
        spatio_output_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_fields,
            hidden_dim,
            patch_size,
            spatial_dims,
            time_steps,
            num_ensemble_members,
            cross_attn_num_heads,
            cross_attn_dropout,
            decoder_num_layers,
            decoder_num_heads,
            decoder_dropout,
        )

        # Spatiotemporal attention after cross-attention
        self.spatiotemporal_attns = nn.ModuleDict(
            {
                field: SpatioTemporalAttention(
                    hidden_dim=hidden_dim,
                    num_heads=spatio_num_heads,
                    attention_dropout=spatio_attn_dropout,
                    dropout=spatio_output_dropout,
                )
                for field in input_fields
            }
        )

    def forward(
        self, x_dict: Dict[str, torch.Tensor], mask_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass includes cross-attention and spatiotemporal attention.

        Args:
            x_dict (dict): Field-to-tensor mapping [B, T, H, W].
            mask_dict (dict, optional): Field-to-mask mapping [B, T, N].

        Returns:
            dict: Ensemble predictions for each field [E, B, T, H, W].
        """
        # === 1) Encode each field ===
        encoded_features = {}
        multi_scale_features = {}
        for field_name, field_data in x_dict.items():
            field_mask = mask_dict[field_name] if (mask_dict and field_name in mask_dict) else None
            encoded, features = self.field_transformers[field_name](field_data, field_mask)
            encoded_features[field_name] = encoded
            multi_scale_features[field_name] = features

        # === 2) Cross-attention among fields ===
        cross_attended_features = {field: feat.clone() for field, feat in encoded_features.items()}
        for f1 in self.input_fields:
            for f2 in self.input_fields:
                if f1 != f2:
                    cross_key = f"{f2}_to_{f1}"
                    cross_attn_layer = self.cross_attentions[cross_key]
                    cross_attended_features[f1] += cross_attn_layer(
                        encoded_features[f1], encoded_features[f2]
                    )

        # === 3) Spatiotemporal attention on cross-attended features ===
        spatiotemporal_features = {}
        for field_name, feat in cross_attended_features.items():
            spatiotemporal_features[field_name] = self.spatiotemporal_attns[field_name](feat)

        # === 4) Decode each field ===
        decoded_features = {}
        for field_name, feat in spatiotemporal_features.items():
            decoded = self.decoders[field_name](feat, multi_scale_features[field_name])
            decoded_features[field_name] = decoded

        # === 5) Ensemble predictions ===
        predictions = {}
        for field_name, field_decoded in decoded_features.items():
            B, T, N, D = field_decoded.shape
            patch_grid = int(math.sqrt(N))
            H = W = patch_grid * self.patch_size

            field_preds = []
            for head in self.ensemble_heads[field_name]:
                pred = head(field_decoded)
                pred = rearrange(pred, "b t (h w) c -> b t (h c) w", h=patch_grid, w=patch_grid)
                pred = F.interpolate(pred, size=(H, W), mode="bilinear")
                pred = rearrange(pred, "b t c w -> b t w c")
                field_preds.append(pred)

            predictions[field_name] = torch.stack(field_preds, dim=0)

        return predictions
