"""Multi-grid Functional Generative Network with a shared latent mesh."""

import torch
from einops import rearrange

from graph_weather.models.fgn.layers.processor import Processor
from graph_weather.models.gencast.graph.graph_builder import GraphBuilder
from graph_weather.models.gencast.layers.decoder import Decoder
from graph_weather.models.gencast.layers.encoder import Encoder


class MultiGridFGN(torch.nn.Module):
    """Encode multiple grids, process their shared mesh representation, and decode.

    Each grid has its own encoder and decoder. All encoders project onto a
    common mesh, their mesh representations are summed, and one processor
    updates that shared representation.

    Args:
        grid_lons: Longitude coordinates for each grid.
        grid_lats: Latitude coordinates for each grid.
        input_features_dims: Number of input features for each grid.
        output_features_dims: Number of output features for each grid.
        noise_dimension: Number of noise features supplied to the processor.
        hidden_dims: Hidden dimensions used by encoders, processor and decoders.
        num_blocks: Number of processor transformer blocks.
        num_heads: Number of attention heads.
        splits: Mesh refinement level used by GraphBuilder.
        num_hops: Number of mesh hops used by the processor graph.
        use_edges_features: Whether to use mesh edge features in the processor.
        scale_factor: Encoder/decoder scale factor.
    """

    def __init__(
        self,
        grid_lons,
        grid_lats,
        input_features_dims,
        output_features_dims,
        noise_dimension,
        hidden_dims,
        num_blocks=4,
        num_heads=4,
        splits=2,
        num_hops=1,
        use_edges_features=True,
        scale_factor=1.0,
    ):
        """Initialize grid-specific encoders and decoders around a shared mesh."""
        super().__init__()

        num_grids = len(grid_lons)
        if num_grids < 2:
            raise ValueError("MultiGridFGN requires at least two grids.")
        if len(grid_lats) != num_grids:
            raise ValueError("Each longitude grid needs a latitude grid.")
        if len(input_features_dims) != num_grids:
            raise ValueError("Each grid needs an input feature dimension.")
        if len(output_features_dims) != num_grids:
            raise ValueError("Each grid needs an output feature dimension.")
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty.")

        self.num_grids = num_grids
        self.grid_shapes = [(len(lon), len(lat)) for lon, lat in zip(grid_lons, grid_lats)]
        self.input_features_dims = list(input_features_dims)
        self.output_features_dims = list(output_features_dims)
        self.noise_dimension = noise_dimension
        self.latent_dim = hidden_dims[-1]
        self.use_edges_features = use_edges_features

        # GraphBuilder creates graph data, but is not itself a torch Module.
        graph_builders = [
            GraphBuilder(
                grid_lon=lon,
                grid_lat=lat,
                splits=splits,
                num_hops=num_hops,
            )
            for lon, lat in zip(grid_lons, grid_lats)
        ]

        # All grid graphs must map onto the same mesh with identical node
        # ordering. Check this before combining their latent representations.
        reference = graph_builders[0]
        self._check_shared_mesh(reference, graph_builders[1:])

        self._register_shared_mesh(reference)

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()

        for i, graphs in enumerate(graph_builders):
            self._register_grid_graph(i, graphs)

            self.encoders.append(
                Encoder(
                    grid_dim=(self.input_features_dims[i] + graphs.grid_nodes_dim),
                    mesh_dim=graphs.mesh_nodes_dim,
                    edge_dim=graphs.g2m_edges_dim,
                    hidden_dims=hidden_dims,
                    activation_layer=torch.nn.SiLU,
                    use_layer_norm=True,
                    scale_factor=scale_factor,
                )
            )
            self.decoders.append(
                Decoder(
                    edges_dim=graphs.m2g_edges_dim,
                    output_dim=self.output_features_dims[i],
                    hidden_dims=hidden_dims,
                    activation_layer=torch.nn.SiLU,
                    use_layer_norm=True,
                )
            )

        self.processor = Processor(
            latent_dim=self.latent_dim,
            edges_dim=(reference.mesh_edges_dim if use_edges_features else None),
            hidden_dims=hidden_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            noise_emb_dim=noise_dimension,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
            sparse=False,
        )

    @staticmethod
    def _check_shared_mesh(reference, others):
        """Reject graphs whose shared mesh topology or features differ."""
        reference_graphs = (
            reference.mesh_graph,
            reference.khop_mesh_graph,
        )

        for other in others:
            for ref_graph, other_graph in zip(
                reference_graphs,
                (other.mesh_graph, other.khop_mesh_graph),
            ):
                for attribute in ("x", "edge_index", "edge_attr"):
                    ref_value = getattr(ref_graph, attribute, None)
                    other_value = getattr(other_graph, attribute, None)

                    if ref_value is None or other_value is None:
                        if ref_value is not other_value:
                            raise ValueError(
                                "All grids must use identical shared mesh " "features and topology."
                            )
                    elif ref_value.shape != other_value.shape or not torch.equal(
                        ref_value, other_value
                    ):
                        raise ValueError(
                            "All grids must use identical shared mesh "
                            "features and topology. Check splits and num_hops."
                        )

    def _register_shared_mesh(self, graphs):
        """Register shared graph tensors so model.to(device) moves them."""
        for prefix, graph in (
            ("mesh", graphs.mesh_graph),
            ("khop_mesh", graphs.khop_mesh_graph),
        ):
            for attribute in ("x", "edge_index", "edge_attr"):
                value = getattr(graph, attribute, None)
                if value is not None:
                    self.register_buffer(
                        f"{prefix}_{attribute}",
                        value,
                        persistent=False,
                    )

    def _register_grid_graph(self, index, graphs):
        """Register graph tensors that belong to one particular grid."""
        entries = (
            (f"g2m_grid_nodes_{index}", graphs.g2m_graph["grid_nodes"].x),
            (f"g2m_mesh_nodes_{index}", graphs.g2m_graph["mesh_nodes"].x),
            (
                f"g2m_edge_attr_{index}",
                graphs.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr,
            ),
            (
                f"g2m_edge_index_{index}",
                graphs.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index,
            ),
            (f"m2g_grid_nodes_{index}", graphs.m2g_graph["grid_nodes"].x),
            (f"m2g_mesh_nodes_{index}", graphs.m2g_graph["mesh_nodes"].x),
            (
                f"m2g_edge_attr_{index}",
                graphs.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_attr,
            ),
            (
                f"m2g_edge_index_{index}",
                graphs.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index,
            ),
        )
        for name, value in entries:
            self.register_buffer(name, value, persistent=False)

    @staticmethod
    def _batch_graph(node_features, edge_index, edge_attr, batch_size):
        """Batch a graph for independent samples in the same forward pass."""
        from torch_geometric.data import Batch, Data

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        batched = Batch.from_data_list([graph] * batch_size)
        return batched.x, batched.edge_index, batched.edge_attr

    def _encode_grid(self, index, grid_features):
        batch_size, num_nodes, _ = grid_features.shape

        grid_nodes = getattr(self, f"g2m_grid_nodes_{index}")
        mesh_nodes = getattr(self, f"g2m_mesh_nodes_{index}")
        edge_attr = getattr(self, f"g2m_edge_attr_{index}")
        edge_index = getattr(self, f"g2m_edge_index_{index}")

        batched_grid, batched_mesh, batched_edges, batched_edge_index = (
            self._batch_heterogeneous_graph(
                grid_nodes,
                mesh_nodes,
                edge_attr,
                edge_index,
                batch_size,
            )
        )

        grid_features = rearrange(grid_features, "b n f -> (b n) f")
        grid_features = torch.cat((grid_features, batched_grid), dim=-1)

        latent_grid, latent_mesh = self.encoders[index](
            grid_features,
            batched_mesh,
            batched_edges,
            batched_edge_index,
        )

        mesh_count = mesh_nodes.shape[0]
        return (
            latent_grid.reshape(batch_size, num_nodes, self.latent_dim),
            latent_mesh.reshape(batch_size, mesh_count, self.latent_dim),
        )

    @staticmethod
    def _batch_heterogeneous_graph(
        grid_nodes,
        mesh_nodes,
        edge_attr,
        edge_index,
        batch_size,
    ):
        """Batch a grid-to-mesh graph without mixing samples."""
        from torch_geometric.data import Batch, HeteroData

        data = HeteroData()
        data["grid_nodes"].x = grid_nodes
        data["mesh_nodes"].x = mesh_nodes
        edge_type = ("grid_nodes", "to", "mesh_nodes")
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr

        batched = Batch.from_data_list([data] * batch_size)
        return (
            batched["grid_nodes"].x,
            batched["mesh_nodes"].x,
            batched[edge_type].edge_attr,
            batched[edge_type].edge_index,
        )

    def _process_mesh(self, latent_mesh, noise_vector):
        batch_size, mesh_count, _ = latent_mesh.shape

        mesh_nodes, edge_index, edge_attr = self._batch_graph(
            self.khop_mesh_x,
            self.khop_mesh_edge_index,
            getattr(self, "khop_mesh_edge_attr", None) if self.use_edges_features else None,
            batch_size,
        )

        latent_mesh = rearrange(latent_mesh, "b n f -> (b n) f")
        repeated_noise = noise_vector.repeat_interleave(mesh_count, dim=0)

        processed = self.processor(
            latent_mesh_nodes=latent_mesh,
            edge_index=edge_index,
            noise_vector=repeated_noise,
            input_edge_attr=edge_attr if self.use_edges_features else None,
        )
        return processed.reshape(batch_size, mesh_count, self.latent_dim)

    def _decode_grid(self, index, latent_mesh, latent_grid):
        batch_size, grid_count, _ = latent_grid.shape

        mesh_nodes = getattr(self, f"m2g_mesh_nodes_{index}")
        grid_nodes = getattr(self, f"m2g_grid_nodes_{index}")
        edge_attr = getattr(self, f"m2g_edge_attr_{index}")
        edge_index = getattr(self, f"m2g_edge_index_{index}")

        from torch_geometric.data import Batch, HeteroData

        data = HeteroData()
        data["mesh_nodes"].x = mesh_nodes
        data["grid_nodes"].x = grid_nodes
        edge_type = ("mesh_nodes", "to", "grid_nodes")
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr
        batched = Batch.from_data_list([data] * batch_size)

        latent_mesh = rearrange(latent_mesh, "b n f -> (b n) f")
        latent_grid = rearrange(latent_grid, "b n f -> (b n) f")

        decoded = self.decoders[index](
            latent_mesh,
            latent_grid,
            batched[edge_type].edge_attr,
            batched[edge_type].edge_index,
        )
        return decoded.reshape(batch_size, grid_count, self.output_features_dims[index])

    @staticmethod
    def _fuse_mesh_latents(mesh_latents):
        """Sum grid-specific encodings on their shared mesh."""
        if not mesh_latents:
            raise ValueError("At least one mesh representation is required.")

        reference_shape = mesh_latents[0].shape
        if any(latent.shape != reference_shape for latent in mesh_latents):
            raise ValueError("All mesh representations must have the same shape to fuse.")
        return torch.stack(mesh_latents, dim=0).sum(dim=0)

    def forward(self, grid_inputs, num_ensemble=2):
        """Return one forecast tensor per grid.

        Each input has shape [batch, longitude, latitude, features].
        Each returned tensor has shape
        [batch, ensemble, longitude, latitude, output_features].
        """
        if len(grid_inputs) != self.num_grids:
            raise ValueError(f"Expected {self.num_grids} grid inputs, got {len(grid_inputs)}.")
        if num_ensemble < 1:
            raise ValueError("num_ensemble must be at least 1.")

        batch_size = grid_inputs[0].shape[0]
        flattened_inputs = []

        for index, grid_input in enumerate(grid_inputs):
            if grid_input.ndim != 4:
                raise ValueError("Each grid input must have shape [batch, lon, lat, features].")
            if grid_input.shape[0] != batch_size:
                raise ValueError("All grid inputs must share a batch size.")

            expected_shape = self.grid_shapes[index]
            if tuple(grid_input.shape[1:3]) != expected_shape:
                raise ValueError(
                    f"Grid {index} has spatial shape {tuple(grid_input.shape[1:3])}; "
                    f"expected {expected_shape}."
                )
            if grid_input.shape[-1] != self.input_features_dims[index]:
                raise ValueError(
                    f"Grid {index} has {grid_input.shape[-1]} input features; "
                    f"expected {self.input_features_dims[index]}."
                )

            flattened_inputs.append(rearrange(grid_input, "b lon lat f -> b (lon lat) f"))

        encoded = [
            self._encode_grid(index, features) for index, features in enumerate(flattened_inputs)
        ]
        latent_grids = [result[0] for result in encoded]
        mesh_latents = [result[1] for result in encoded]
        fused_mesh = self._fuse_mesh_latents(mesh_latents)

        outputs = [[] for _ in range(self.num_grids)]
        for _ in range(num_ensemble):
            noise = torch.randn(
                batch_size,
                self.noise_dimension,
                device=fused_mesh.device,
                dtype=fused_mesh.dtype,
            )
            processed_mesh = self._process_mesh(fused_mesh, noise)

            for index, latent_grid in enumerate(latent_grids):
                decoded = self._decode_grid(index, processed_mesh, latent_grid)
                lon_count, lat_count = self.grid_shapes[index]
                decoded = decoded.reshape(
                    batch_size,
                    lon_count,
                    lat_count,
                    self.output_features_dims[index],
                )
                outputs[index].append(decoded)

        return [torch.stack(grid_outputs, dim=1) for grid_outputs in outputs]
