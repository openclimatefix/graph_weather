"""Processor layer.

The processor:
- compute a sequence of transformer blocks applied to the mesh.
- condition on the noise level.
"""

import torch

from graph_weather.models.gencast.layers.modules import MLP, CondTransformerBlock, FourierEmbedding

try:
    from graph_weather.models.gencast.layers.experimental import SparseTransformer

    has_dgl = True
except ImportError:
    has_dgl = False


class Processor(torch.nn.Module):
    """GenCast's Processor

    The Processor is a sequence of transformer blocks conditioned on noise level. If the graph has
    many edges, setting sparse=True may perform better in terms of memory and speed. Note that
    sparse=False uses PyG as the backend, while sparse=True uses DGL. The two implementations are
    not exactly equivalent: the former is described in the paper "Masked Label Prediction: Unified
    Message Passing Model for Semi-Supervised Classification" and can also handle edge features,
    while the latter is a classical transformer that performs multi-head attention utilizing the
    mask's sparsity and does not include edge features in the computations.

    Note: The GenCast paper does not provide specific details regarding the implementation of the
    transformer architecture for graphs.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        num_blocks: int,
        num_heads: int,
        num_frequencies: int,
        base_period: int,
        noise_emb_dim: int,
        edges_dim: int | None = None,
        activation_layer: torch.nn.Module = torch.nn.ReLU,
        use_layer_norm: bool = True,
        sparse: bool = False,
    ):
        """Initialize the Processor.

        Args:
            latent_dim (int): dimension of nodes' features.
            hidden_dims (list[int]): hidden dimensions of internal MLPs.
            num_blocks (int): number of transformer blocks.
            num_heads (int): number of heads for multi-head attention.
            num_frequencies (int): number of frequencies for the noise Fourier embedding.
            base_period (int): base period for the noise Fourier embedding.
            noise_emb_dim (int): dimension of output of noise embedding.
            edges_dim (int, optional): dimension of edges' features. If None does not uses edges
                features in TransformerConv. Defaults to None.
            activation_layer (torch.nn.Module): activation function of internal MLPs.
                Defaults to torch.nn.ReLU.
            use_layer_norm (bool): if true add a LayerNorm at the end of the embedding MLP.
                Defaults to True.
            sparse (bool): if true use DGL as backend (experimental). Defaults to False.
        """
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim % num_heads != 0:
            raise ValueError("The latent dimension should be divisible by the number of heads.")

        # Embedders
        self.fourier_embedder = FourierEmbedding(
            output_dim=noise_emb_dim, num_frequencies=num_frequencies, base_period=base_period
        )

        self.edges_dim = edges_dim
        if edges_dim is not None:
            self.edges_mlp = MLP(
                input_dim=edges_dim,
                hidden_dims=hidden_dims,
                activation_layer=activation_layer,
                use_layer_norm=use_layer_norm,
                bias=True,
                activate_final=False,
            )

        # Tranformers Blocks
        self.cond_transformers = torch.nn.ModuleList()
        if not sparse:
            for _ in range(num_blocks - 1):
                # concatenating multi-head attention
                self.cond_transformers.append(
                    CondTransformerBlock(
                        conditioning_dim=noise_emb_dim,
                        input_dim=latent_dim,
                        output_dim=latent_dim // num_heads,
                        edges_dim=hidden_dims[-1] if (edges_dim is not None) else None,
                        num_heads=num_heads,
                        concat=True,
                        beta=True,
                        activation_layer=activation_layer,
                    )
                )

            # averaging multi-head attention
            self.cond_transformers.append(
                CondTransformerBlock(
                    conditioning_dim=noise_emb_dim,
                    input_dim=latent_dim,
                    output_dim=latent_dim,
                    edges_dim=hidden_dims[-1] if (edges_dim is not None) else None,
                    num_heads=num_heads,
                    concat=False,
                    beta=True,
                    activation_layer=None,
                )
            )
        else:
            if not has_dgl:
                raise ValueError("Please install DGL to use sparsity.")

            for _ in range(num_blocks - 1):
                # concatenating multi-head attention
                self.cond_transformers.append(
                    SparseTransformer(
                        conditioning_dim=noise_emb_dim,
                        input_dim=latent_dim,
                        output_dim=latent_dim,
                        num_heads=num_heads,
                        activation_layer=activation_layer,
                    )
                )
                # do we really need averaging for last block?

    def _check_args(self, latent_mesh_nodes, noise_levels, input_edge_attr):
        if not latent_mesh_nodes.shape[-1] == self.latent_dim:
            raise ValueError(
                "The dimension of the mesh nodes is different from the latent dimension provided at"
                " initialization."
            )

        if not latent_mesh_nodes.shape[0] == noise_levels.shape[0]:
            raise ValueError(
                "The number of noise levels and mesh nodes should be the same, but got "
                f"{latent_mesh_nodes.shape[0]} and {noise_levels.shape[0]}. Eventually repeat the "
                " noise level for each node in the same batch."
            )

        if (input_edge_attr is not None) and (self.edges_dim is None):
            raise ValueError("To use input_edge_attr initialize the processor with edges_dim.")

    def forward(
        self,
        latent_mesh_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        noise_levels: torch.Tensor,
        input_edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            latent_mesh_nodes (torch.Tensor): mesh nodes' features.
            edge_index (torch.Tensor): edge index tensor.
            noise_levels (torch.Tensor): log-noise levels.
            input_edge_attr (torch.Tensor, optional): mesh edges' features.

        Returns:
            torch.Tensor: latent mesh nodes.
        """
        self._check_args(latent_mesh_nodes, noise_levels, input_edge_attr)

        # embedding
        noise_emb = self.fourier_embedder(noise_levels)

        if self.edges_dim is not None:
            edges_emb = self.edges_mlp(input_edge_attr)
        else:
            edges_emb = None

        # apply transformer blocks
        for cond_transformer in self.cond_transformers:
            latent_mesh_nodes = cond_transformer(
                x=latent_mesh_nodes,
                edge_index=edge_index,
                cond_param=noise_emb,
                edge_attr=edges_emb,
            )

        return latent_mesh_nodes
