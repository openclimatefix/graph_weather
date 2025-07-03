import einops
import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin

from graph_weather.models.fgn.layers.processor import Processor
from graph_weather.models.gencast.graph.graph_builder import GraphBuilder
from graph_weather.models.gencast.layers.decoder import Decoder
from graph_weather.models.gencast.layers.encoder import Encoder
from graph_weather.models.gencast.utils.batching import batch, hetero_batch


class FunctionalGenerativeNetwork(torch.nn.Module, PyTorchModelHubMixin):
    """Functional Generative Network (FGN) for weather prediction.

    This class defines a generative model that predicts future weather states
    based on previous observations and noise levels.
    """

    def __init__(
        self,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        input_features_dim: int,
        output_features_dim: int,
        noise_dimension: int,
        hidden_dims: list[int] = [768, 768],
        num_blocks: int = 24,
        num_heads: int = 4,
        splits: int = 6,
        num_hops: int = 6,
        device: torch.device = torch.device("cpu"),
        sparse: bool = False,
        use_edges_features: bool = True,
        scale_factor: float = 1.0,
    ):
        """Initialize the FGN.

        Args:
            grid_lon (np.ndarray): array of longitudes.
            grid_lat (np.ndarray): array of latitudes.
            input_features_dim (int): dimension of the input features for a single timestep.
            output_features_dim (int): dimension of the target features.
            noise_dimension (int): dimension of the noise vector used for conditioning the model.
            hidden_dims (list[int], optional): list of dimensions for the hidden layers in the MLPs
                used in GenCast. This also determines the latent dimension. Defaults to [512, 512].
            num_blocks (int, optional): number of transformer blocks in Processor. Defaults to 16.
            num_heads (int, optional): number of heads for each transformer. Defaults to 4.
            splits (int, optional): number of time to split the icosphere during graph building.
                Defaults to 6.
            num_hops (int, optional): the transformes will attention to the (2^num_hops)-neighbours
                of each node. Defaults to 6.
            device (torch.device, optional): device on which we want to build graph.
                Defaults to torch.device("cpu").
            sparse (bool): if true the processor will apply Sparse Attention using DGL backend.
                Defaults to False.
            use_edges_features (bool): if true use mesh edges features inside the Processor.
                Defaults to True.
            scale_factor (float):  in the Encoder the message passing output is multiplied by the
                scale factor. This is important when you want to fine-tune a pretrained model to a
                higher resolution. Defaults to 1.
        """
        super().__init__()
        self.num_lon = len(grid_lon)
        self.num_lat = len(grid_lat)
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.use_edges_features = use_edges_features
        self.noise_dimension = noise_dimension

        # Initialize graph
        self.graphs = GraphBuilder(
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            splits=splits,
            num_hops=num_hops,
            device=device,
            add_edge_features_to_khop=use_edges_features,
        )

        self._register_graph()

        # Initialize Encoder
        self.encoder = Encoder(
            grid_dim=input_features_dim + self.graphs.grid_nodes_dim,
            mesh_dim=self.graphs.mesh_nodes_dim,
            edge_dim=self.graphs.g2m_edges_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
            scale_factor=scale_factor,
        )

        # Initialize Processor
        if sparse and use_edges_features:
            raise ValueError("Sparse processor don't support edges features.")

        self.processor = Processor(
            latent_dim=hidden_dims[-1],
            edges_dim=self.graphs.mesh_edges_dim if use_edges_features else None,
            hidden_dims=hidden_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            noise_emb_dim=noise_dimension,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
            sparse=sparse,
        )

        # Initialize Decoder
        self.decoder = Decoder(
            edges_dim=self.graphs.m2g_edges_dim,
            output_dim=output_features_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
        )

    def _run_encoder(self, grid_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = grid_features.shape[0]
        batched_senders, batched_receivers, batched_edge_index, batched_edge_attr = hetero_batch(
            self.g2m_grid_nodes,
            self.g2m_mesh_nodes,
            self.g2m_edge_index,
            self.g2m_edge_attr,
            batch_size,
        )
        # load features.
        grid_features = einops.rearrange(grid_features, "b n f -> (b n) f")
        input_grid_nodes = torch.cat([grid_features, batched_senders], dim=-1)
        input_mesh_nodes = batched_receivers
        input_edge_attr = batched_edge_attr
        edge_index = batched_edge_index

        # run the encoder.
        latent_grid_nodes, latent_mesh_nodes = self.encoder(
            input_grid_nodes=input_grid_nodes,
            input_mesh_nodes=input_mesh_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
        )

        # restore nodes dimension: [b, n, f]
        latent_grid_nodes = einops.rearrange(latent_grid_nodes, "(b n) f -> b n f", b=batch_size)
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(latent_grid_nodes).any()
        assert not torch.isnan(latent_mesh_nodes).any()
        return latent_grid_nodes, latent_mesh_nodes

    def _run_decoder(
        self, latent_mesh_nodes: torch.Tensor, latent_grid_nodes: torch.Tensor
    ) -> torch.Tensor:
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = latent_mesh_nodes.shape[0]
        _, _, batched_edge_index, batched_edge_attr = hetero_batch(
            self.m2g_mesh_nodes,
            self.m2g_grid_nodes,
            self.m2g_edge_index,
            self.m2g_edge_attr,
            batch_size,
        )

        # load features.
        input_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
        input_grid_nodes = einops.rearrange(latent_grid_nodes, "b n f -> (b n) f")
        input_edge_attr = batched_edge_attr
        edge_index = batched_edge_index

        # run the decoder.
        output_grid_nodes = self.decoder(
            input_mesh_nodes=input_mesh_nodes,
            input_grid_nodes=input_grid_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
        )

        # restore nodes dimension: [b, n, f]
        output_grid_nodes = einops.rearrange(output_grid_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(output_grid_nodes).any()
        return output_grid_nodes

    def _run_processor(
        self, latent_mesh_nodes: torch.Tensor, noise_vectors: torch.Tensor
    ) -> torch.Tensor:
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = latent_mesh_nodes.shape[0]
        num_nodes = latent_mesh_nodes.shape[1]
        _, batched_edge_index, batched_edge_attr = batch(
            self.khop_mesh_nodes,
            self.khop_mesh_edge_index,
            self.khop_mesh_edge_attr if self.use_edges_features else None,
            batch_size,
        )

        # load features.
        # TODO Add Sin/cos day of year here to the features
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
        input_edge_attr = batched_edge_attr
        edge_index = batched_edge_index

        # repeat noise levels for each node.
        noise_vectors = einops.repeat(noise_vectors, "b f -> (b n) f", n=num_nodes)

        # run the processor.
        latent_mesh_nodes = self.processor.forward(
            latent_mesh_nodes=latent_mesh_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
            noise_vector=noise_vectors,
        )

        # restore nodes dimension: [b, n, f]
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(latent_mesh_nodes).any()
        return latent_mesh_nodes

    def _register_graph(self):
        # we need to register all the tensors associated with the graph as buffers. In this way they
        # will move to the same device of the model. These tensors won't be part of the state since
        # persistent is set to False.

        self.register_buffer(
            "g2m_grid_nodes", self.graphs.g2m_graph["grid_nodes"].x, persistent=False
        )
        self.register_buffer(
            "g2m_mesh_nodes", self.graphs.g2m_graph["mesh_nodes"].x, persistent=False
        )
        self.register_buffer(
            "g2m_edge_attr",
            self.graphs.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr,
            persistent=False,
        )
        self.register_buffer(
            "g2m_edge_index",
            self.graphs.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index,
            persistent=False,
        )

        self.register_buffer("mesh_nodes", self.graphs.mesh_graph.x, persistent=False)
        self.register_buffer("mesh_edge_attr", self.graphs.mesh_graph.edge_attr, persistent=False)
        self.register_buffer("mesh_edge_index", self.graphs.mesh_graph.edge_index, persistent=False)

        self.register_buffer("khop_mesh_nodes", self.graphs.khop_mesh_graph.x, persistent=False)
        self.register_buffer(
            "khop_mesh_edge_attr", self.graphs.khop_mesh_graph.edge_attr, persistent=False
        )
        self.register_buffer(
            "khop_mesh_edge_index", self.graphs.khop_mesh_graph.edge_index, persistent=False
        )

        self.register_buffer(
            "m2g_grid_nodes", self.graphs.m2g_graph["grid_nodes"].x, persistent=False
        )
        self.register_buffer(
            "m2g_mesh_nodes", self.graphs.m2g_graph["mesh_nodes"].x, persistent=False
        )
        self.register_buffer(
            "m2g_edge_attr",
            self.graphs.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_attr,
            persistent=False,
        )
        self.register_buffer(
            "m2g_edge_index",
            self.graphs.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index,
            persistent=False,
        )

    def forward(self, previous_weather_state: torch.Tensor, num_ensemble: int = 2) -> torch.Tensor:
        """
        Predict the next weather state given the previous weather state

        Run multiple predictions on the same inputs with different noise vectors to get an ensemble

        Args:
            previous_weather_state: Torch tensor
                The previous weather state, shape (batch_size, num_channels, height, width).
            num_ensemble: number of ensemble predictions to make, default is 2

        Returns:
            torch.Tensor: The predicted future weather state, shape (batch_size, num_ensemble, num_channels, height, width).
        """

        previous_weather_state = einops.rearrange(
            previous_weather_state, "b lon lat f -> b (lon lat) f"
        )

        predictions = []
        for ensemble in range(num_ensemble):
            noise_vector = torch.randn(
                (previous_weather_state.shape[0], self.noise_dimension),
                device=previous_weather_state.device,
            )
            # TODO Append in the sin/cos day of year here to the encoded state
            # TODO Processor is only one with the conditional state
            latent_grid_nodes, latent_mesh_nodes = self._run_encoder(previous_weather_state)
            latent_mesh_nodes = self._run_processor(latent_mesh_nodes, noise_vector)
            out = self._run_decoder(latent_mesh_nodes, latent_grid_nodes)
            # restore lon/lat dimensions.
            prediction = einops.rearrange(out, "b (lon lat) f -> b lon lat f", lon=self.num_lon)
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)
