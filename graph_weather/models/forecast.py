"""Model for forecasting weather from NWP states"""


import torch
from huggingface_hub import PyTorchModelHubMixin

from graph_weather.models import Decoder, Encoder, Processor


class GraphWeatherForecaster(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        lat_lons: list,
        resolution: int = 2,
        feature_dim: int = 78,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
        norm_type="LayerNorm",
    ):
        super().__init__()

        self.encoder = Encoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=feature_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
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
            input_dim=feature_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            hidden_layers_decoder=hidden_layers_decoder,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons

        Returns:
            The next state in the forecast
        """
        x, edge_idx, edge_attr = self.encoder(features)
        out = self.processor(x, edge_idx, edge_attr)
        pred = self.decoder(out, features)
        return pred
