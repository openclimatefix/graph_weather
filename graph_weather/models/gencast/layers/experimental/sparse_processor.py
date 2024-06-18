"""Sparse Processor layer.

This implementation is experimental and uses DGL_sparse as backend.

The processor:
- compute a sequence of transformer blocks applied to the mesh.
- condition on the noise level.
"""

import dgl.sparse as dglsp
import torch

from graph_weather.models.gencast.layers.experimental.sparse_transformer import SparseTransformer
from graph_weather.models.gencast.layers.modules import FourierEmbedding


class SparseProcessor(torch.nn.Module):
    """GenCast's Sparse Processor."""

    def __init__(
        self,
        latent_dim: int,
        num_blocks: int,
        num_heads: int,
        num_frequencies: int,
        base_period: int,
        noise_emb_dim: int,
        activation_layer: torch.nn.Module = torch.nn.ReLU,
    ):
        """Initialize the Processor.

        Args:
            latent_dim (int): dimension of nodes' features.
            num_blocks (int): number of transformer blocks.
            num_heads (int): number of heads for multi-head attention.
            num_frequencies (int): number of frequencies for the noise Fourier embedding.
            base_period (int): base period for the noise Fourier embedding.
            noise_emb_dim (int): dimension of output of noise embedding.
            activation_layer (torch.nn.Module): activation function of internal MLPs.
                Defaults to torch.nn.ReLU.
        """
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim % num_heads != 0:
            raise ValueError("The latent dimension should be divisible by the number of heads.")

        # Embedders
        self.fourier_embedder = FourierEmbedding(
            output_dim=noise_emb_dim, num_frequencies=num_frequencies, base_period=base_period
        )

        # Tranformers Blocks
        self.cond_transformers = torch.nn.ModuleList()

        for _ in range(num_blocks):
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

    def forward(
        self,
        latent_mesh_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        noise_levels: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            latent_mesh_nodes (torch.Tensor): mesh nodes' features.
            edge_index (torch.Tensor): edge index tensor.
            noise_levels (torch.Tensor): log-noise levels.
            *args: they will be ignored by the model
            **kwargs: they will be ignored by the model.

        Returns:
            torch.Tensor: latent mesh nodes.
        """
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

        # Embedding
        noise_emb = self.fourier_embedder(noise_levels)

        for cond_transformer in self.cond_transformers:
            latent_mesh_nodes = cond_transformer(
                x=latent_mesh_nodes,
                edge_index=edge_index,
                cond_param=noise_emb,
            )

        return latent_mesh_nodes
