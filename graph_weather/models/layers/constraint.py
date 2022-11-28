"""

"""
import torch


class Constraint(torch.nn.Module):
    def __init__(
        self,
        lat_lons,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        output_edge_dim: int = 256,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        use_checkpointing: bool = False,
    ):
        super().__init__(
            lat_lons,
            resolution,
            input_dim,
            output_dim,
            output_edge_dim,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
            hidden_dim_decoder,
            hidden_layers_decoder,
            use_checkpointing,
        )

    def forward(
        self, processor_features: torch.Tensor, start_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Constrains output from previous layer

        Args:
            processor_features: Processed features in shape [B*Nodes, Features]
            start_features: Original input features to the encoder, with shape [B, Nodes, Features]

        Returns:
            Updated features for model
        """
        out = super().forward(processor_features, start_features.shape[0])
        out = out + start_features  # residual connection
        return out
