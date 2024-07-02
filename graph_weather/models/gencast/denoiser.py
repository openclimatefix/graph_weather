"""Denoiser.

The denoiser takes as inputs the previous two timesteps, the corrupted target residual, and the
noise level, and outputs the denoised predictions. It performs the following tasks:
1. Initializes the graph, encoder, processor, and decoder.
2. Computes f_theta as the combination of encoder, processor, and decoder.
3. Preconditions f_theta on the noise levels using the parametrization from Karras et al. (2022).
"""

import einops
import numpy as np
import torch
from torch_geometric.data import Batch

from graph_weather.models.gencast.graph.graph_builder import GraphBuilder
from graph_weather.models.gencast.layers.decoder import Decoder
from graph_weather.models.gencast.layers.encoder import Encoder
from graph_weather.models.gencast.layers.processor import Processor
from graph_weather.models.gencast.utils.noise import Preconditioner


class Denoiser(torch.nn.Module):
    """GenCast's Denoiser."""

    def __init__(
        self,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        input_features_dim: int,
        output_features_dim: int,
        hidden_dims: list[int] = [512, 512],
        num_blocks: int = 16,
        num_heads: int = 4,
        splits: int = 6,
        num_hops: int = 6,
        device: torch.device = torch.device("cpu"),
        sparse: bool = False,
        use_edges_features: bool = True,
    ):
        """Initialize the Denoiser.

        Args:
            grid_lon (np.ndarray): array of longitudes.
            grid_lat (np.ndarray): array of latitudes.
            input_features_dim (int): dimension of the input features for a single timestep.
            output_features_dim (int): dimension of the target features.
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
        """
        super().__init__()
        self.num_lon = len(grid_lon)
        self.num_lat = len(grid_lat)
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.use_edges_features = use_edges_features

        # Initialize graph
        self.graphs = GraphBuilder(
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            splits=splits,
            num_hops=num_hops,
            device=device,
            add_edge_features_to_khop=use_edges_features,
        )

        # Initialize Encoder
        self.encoder = Encoder(
            grid_dim=output_features_dim + 2 * input_features_dim + self.graphs.grid_nodes_dim,
            mesh_dim=self.graphs.mesh_nodes_dim,
            edge_dim=self.graphs.g2m_edges_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
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
            num_frequencies=32,
            base_period=16,
            noise_emb_dim=16,
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

        # Initialize preconditioning functions
        self.precs = Preconditioner(sigma_data=1.0)

    def _check_shapes(self, corrupted_targets, prev_inputs, noise_levels):
        batch_size = prev_inputs.shape[0]
        exp_inputs_shape = (batch_size, self.num_lon, self.num_lat, 2 * self.input_features_dim)
        exp_targets_shape = (batch_size, self.num_lon, self.num_lat, self.output_features_dim)
        exp_noise_shape = (batch_size, 1)

        if not all(
            [
                corrupted_targets.shape == exp_targets_shape,
                prev_inputs.shape == exp_inputs_shape,
                noise_levels.shape == exp_noise_shape,
            ]
        ):
            raise ValueError(
                "The shapes of the input tensors don't match with the initialization parameters: "
                f"expected {exp_inputs_shape} for prev_inputs, {exp_targets_shape} for targets and "
                f"{exp_noise_shape} for noise_levels."
            )

    def _run_encoder(self, grid_features):
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = grid_features.shape[0]
        g2m_batched = Batch.from_data_list([self.graphs.g2m_graph] * batch_size)

        # load features.
        grid_features = einops.rearrange(grid_features, "b n f -> (b n) f")
        input_grid_nodes = torch.cat([grid_features, g2m_batched["grid_nodes"].x], dim=-1).type(
            torch.float32
        )
        input_mesh_nodes = g2m_batched["mesh_nodes"].x
        input_edge_attr = g2m_batched["grid_nodes", "to", "mesh_nodes"].edge_attr
        edge_index = g2m_batched["grid_nodes", "to", "mesh_nodes"].edge_index

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

    def _run_decoder(self, latent_mesh_nodes, latent_grid_nodes):
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = latent_mesh_nodes.shape[0]
        m2g_batched = Batch.from_data_list([self.graphs.m2g_graph] * batch_size)

        # load features.
        input_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
        input_grid_nodes = einops.rearrange(latent_grid_nodes, "b n f -> (b n) f")
        input_edge_attr = m2g_batched["mesh_nodes", "to", "grid_nodes"].edge_attr
        edge_index = m2g_batched["mesh_nodes", "to", "grid_nodes"].edge_index

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

    def _run_processor(self, latent_mesh_nodes, noise_levels):
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = latent_mesh_nodes.shape[0]
        num_nodes = latent_mesh_nodes.shape[1]
        mesh_batched = Batch.from_data_list([self.graphs.khop_mesh_graph] * batch_size)

        # load features.
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
        input_edge_attr = mesh_batched.edge_attr if self.use_edges_features else None
        edge_index = mesh_batched.edge_index

        # repeat noise levels for each node.
        noise_levels = einops.repeat(noise_levels, "b f -> (b n) f", n=num_nodes)

        # run the processor.
        latent_mesh_nodes = self.processor.forward(
            latent_mesh_nodes=latent_mesh_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
            noise_levels=noise_levels,
        )

        # restore nodes dimension: [b, n, f]
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(latent_mesh_nodes).any()
        return latent_mesh_nodes

    def _f_theta(self, grid_features, noise_levels):
        # run encoder, processor and decoder.
        latent_grid_nodes, latent_mesh_nodes = self._run_encoder(grid_features)
        latent_mesh_nodes = self._run_processor(latent_mesh_nodes, noise_levels)
        output_grid_nodes = self._run_decoder(latent_mesh_nodes, latent_grid_nodes)
        return output_grid_nodes

    def forward(
        self, corrupted_targets: torch.Tensor, prev_inputs: torch.Tensor, noise_levels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the denoiser output.

        The denoiser is a version of the (encoder, processor, decoder)-model (called f_theta),
        preconditioned on the noise levels, as described below:

        D(Z, X, sigma) := c_skip(sigma)Z + c_out(sigma) * f_theta(c_in(sigma)Z, X, c_noise(sigma)),

        where Z is the corrupted target, X is the previous two timesteps concatenated and sigma is
        the noise level used for Z's corruption.

        Args:
            corrupted_targets (torch.Tensor): the target residuals corrupted by noise.
            prev_inputs (torch.Tensor): the previous two timesteps concatenated across the features'
                dimension.
            noise_levels (torch.Tensor): the noise level used for corruption.
        """
        # check shapes and noise.
        self._check_shapes(corrupted_targets, prev_inputs, noise_levels)
        if not (noise_levels > 0).any():
            raise ValueError("All the noise levels must be strictly positive.")

        # flatten lon/lat dimensions.
        prev_inputs = einops.rearrange(prev_inputs, "b lon lat f -> b (lon lat) f")
        corrupted_targets = einops.rearrange(corrupted_targets, "b lon lat f -> b (lon lat) f")

        # apply preconditioning functions to target and noise.
        scaled_targets = self.precs.c_in(noise_levels)[:, :, None] * corrupted_targets
        scaled_noise_levels = self.precs.c_noise(noise_levels)

        # concatenate inputs and targets across features dimension.
        grid_features = torch.cat((scaled_targets, prev_inputs), dim=-1)

        # run the model.
        preds = self._f_theta(grid_features, scaled_noise_levels)

        # add skip connection.
        out = (
            self.precs.c_skip(noise_levels)[:, :, None] * corrupted_targets
            + self.precs.c_out(noise_levels)[:, :, None] * preds
        )

        # restore lon/lat dimensions.
        out = einops.rearrange(out, "b (lon lat) f -> b lon lat f", lon=self.num_lon)
        return out
