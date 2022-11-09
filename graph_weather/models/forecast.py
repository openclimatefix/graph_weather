"""Model for forecasting weather from NWP states"""
import torch
from typing import Optional
from huggingface_hub import PyTorchModelHubMixin

from graph_weather.models import Decoder, Encoder, Processor


class GraphWeatherForecaster(torch.nn.Module, PyTorchModelHubMixin):
    """Main weather prediction model from the paper"""

    def __init__(
        self,
        lat_lons: list,
        resolution: int = 2,
        feature_dim: int = 78,
        aux_dim: int = 24,
        output_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            output_dim: Optional, output feature size, useful if want only subset of variables in output
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Use gradient checkpointing to reduce model memory
        """
        super().__init__()
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim

        self.encoder = Encoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=feature_dim + aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            use_checkpointing=use_checkpointing,
        )
        self.processor = Processor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
        )
        self.decoder = Decoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        x, edge_idx, edge_attr = self.encoder(features)
        x = self.processor(x, edge_idx, edge_attr)
        x = self.decoder(x, features[..., : self.feature_dim])
        return x
