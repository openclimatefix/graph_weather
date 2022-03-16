"""Model for forecasting weather from NWP states"""
import torch
from huggingface_hub import PyTorchModelHubMixin

from graph_weather.models import AssimilatorDecoder, AssimilatorEncoder, Processor


class GraphWeatherAssimilator(torch.nn.Module, PyTorchModelHubMixin):
    """Model to generate analysis file from raw observations"""

    def __init__(
        self,
        output_lat_lons: list,
        resolution: int = 2,
        observation_dim: int = 2,
        analysis_dim: int = 78,
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
        """
        Graph Weather Data Assimilation model

        Args:
        observation_lat_lons: Lat/lon points of the observations
            output_lat_lons: List of latitude and longitudes for the output analysis
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            observation_dim: Input feature size
            analysis_dim: Output Analysis feature dim
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
        """
        super().__init__()

        self.encoder = AssimilatorEncoder(
            resolution=resolution,
            input_dim=observation_dim,
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
        self.decoder = AssimilatorDecoder(
            lat_lons=output_lat_lons,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=analysis_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            hidden_layers_decoder=hidden_layers_decoder,
        )

    def forward(self, features: torch.Tensor, obs_lat_lon_heights: torch.Tensor) -> torch.Tensor:
        """
        Compute the analysis output

        Args:
            features: The input features, aligned with the order of lat_lons_heights
            obs_lat_lon_heights: Observation lat/lon/heights in same order as features

        Returns:
            The next state in the forecast
        """
        x, edge_idx, edge_attr = self.encoder(features, obs_lat_lon_heights)
        x = self.processor(x, edge_idx, edge_attr)
        x = self.decoder(x, features.shape[0])
        return x
