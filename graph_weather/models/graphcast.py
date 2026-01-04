"""GraphCast model with hierarchical gradient checkpointing.

This module provides a complete GraphCast-style weather forecasting model
with NVIDIA-style hierarchical gradient checkpointing for memory-efficient training.

Based on:
- "Forecasting Global Weather with Graph Neural Networks" (arXiv:2202.07575)
- NVIDIA PhysicsNeMo GraphCast implementation
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from graph_weather.models.layers.decoder import Decoder
from graph_weather.models.layers.encoder import Encoder
from graph_weather.models.layers.processor import Processor


class GraphCast(torch.nn.Module):
    """GraphCast model with hierarchical gradient checkpointing.

    This model combines Encoder, Processor, and Decoder with NVIDIA-style
    hierarchical checkpointing controls for flexible memory-compute tradeoffs.

    Parameters
    ----------
    lat_lons : list
        List of (lat, lon) tuples defining the grid points
    resolution : int, optional
        H3 resolution level, by default 2
    input_dim : int, optional
        Input feature dimension, by default 78
    output_dim : int, optional
        Output feature dimension, by default 78
    hidden_dim : int, optional
        Hidden dimension for all layers, by default 256
    num_processor_blocks : int, optional
        Number of message passing blocks in processor, by default 9
    hidden_layers : int, optional
        Number of hidden layers in MLPs, by default 2
    mlp_norm_type : str, optional
        Normalization type for MLPs, by default "LayerNorm"
    use_checkpointing : bool, optional
        Enable fine-grained checkpointing in all layers, by default False
    efficient_batching : bool, optional
        Use efficient batching (avoid graph replication), by default False

    Hierarchical Checkpointing Methods
    -----------------------------------
    set_checkpoint_model(flag: bool)
        Checkpoint entire forward pass as single segment.
        Disables all other checkpointing when enabled.
        Maximum memory savings, highest recomputation cost.

    set_checkpoint_encoder(flag: bool)
        Checkpoint encoder section (embedder + encoder + first processor).

    set_checkpoint_processor(segments: int)
        Checkpoint processor with configurable segments:
        - segments = 0: No checkpointing
        - segments = -1: Checkpoint entire processor as one block
        - segments = N: Checkpoint every N blocks (not yet implemented)

    set_checkpoint_decoder(flag: bool)
        Checkpoint decoder section (last processor + decoder).

    Examples
    --------
    >>> # Create model with hierarchical checkpointing
    >>> lat_lons = [(lat, lon) for lat in range(-90, 90, 5)
    ...             for lon in range(0, 360, 5)]
    >>> model = GraphCast(lat_lons, use_checkpointing=False)
    >>>
    >>> # Strategy 1: Checkpoint entire model (maximum memory savings)
    >>> model.set_checkpoint_model(True)
    >>>
    >>> # Strategy 2: Fine-grained checkpointing
    >>> model.set_checkpoint_model(False)
    >>> model.set_checkpoint_encoder(True)
    >>> model.set_checkpoint_processor(-1)  # Checkpoint all processor blocks
    >>> model.set_checkpoint_decoder(True)
    >>>
    >>> # Strategy 3: No checkpointing (maximum speed)
    >>> model.set_checkpoint_model(False)
    >>> model.set_checkpoint_encoder(False)
    >>> model.set_checkpoint_processor(0)
    >>> model.set_checkpoint_decoder(False)
    """

    def __init__(
        self,
        lat_lons: list,
        resolution: int = 2,
        input_dim: int = 78,
        output_dim: int = 78,
        hidden_dim: int = 256,
        num_processor_blocks: int = 9,
        hidden_layers: int = 2,
        mlp_norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        efficient_batching: bool = False,
    ):
        super().__init__()

        self.lat_lons = lat_lons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.efficient_batching = efficient_batching

        # Initialize components
        self.encoder = Encoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=input_dim,
            output_dim=hidden_dim,
            output_edge_dim=hidden_dim,
            hidden_dim_processor_node=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
            hidden_layers_processor_node=hidden_layers,
            hidden_layers_processor_edge=hidden_layers,
            mlp_norm_type=mlp_norm_type,
            use_checkpointing=use_checkpointing,
            efficient_batching=efficient_batching,
        )

        self.processor = Processor(
            input_dim=hidden_dim,
            edge_dim=hidden_dim,
            num_blocks=num_processor_blocks,
            hidden_dim_processor_node=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
            hidden_layers_processor_node=hidden_layers,
            hidden_layers_processor_edge=hidden_layers,
            mlp_norm_type=mlp_norm_type,
            use_checkpointing=use_checkpointing,
        )

        self.decoder = Decoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim_processor_node=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
            hidden_layers_processor_node=hidden_layers,
            hidden_layers_processor_edge=hidden_layers,
            mlp_norm_type=mlp_norm_type,
            hidden_dim_decoder=hidden_dim,
            hidden_layers_decoder=hidden_layers,
            use_checkpointing=use_checkpointing,
            efficient_batching=efficient_batching,
        )

        # Hierarchical checkpointing flags (default: use fine-grained checkpointing)
        self._checkpoint_model = False
        self._checkpoint_encoder = False
        self._checkpoint_processor_segments = 0  # 0 = use layer's internal checkpointing
        self._checkpoint_decoder = False

    def set_checkpoint_model(self, checkpoint_flag: bool):
        """Checkpoint entire model as a single segment.

        When enabled, creates one checkpoint for the entire forward pass.
        This provides maximum memory savings but highest recomputation cost.
        Disables all other hierarchical checkpointing when enabled.

        Parameters
        ----------
        checkpoint_flag : bool
            If True, checkpoint entire model. If False, use hierarchical checkpointing.
        """
        self._checkpoint_model = checkpoint_flag
        if checkpoint_flag:
            # Disable all fine-grained checkpointing
            self._checkpoint_encoder = False
            self._checkpoint_processor_segments = 0
            self._checkpoint_decoder = False
            # Disable layer-level checkpointing
            self._set_layer_checkpointing(False)

    def set_checkpoint_encoder(self, checkpoint_flag: bool):
        """Checkpoint encoder section.

        Checkpoints the encoder forward pass as a single segment.
        Only effective when set_checkpoint_model(False).

        Parameters
        ----------
        checkpoint_flag : bool
            If True, checkpoint encoder section.
        """
        self._checkpoint_encoder = checkpoint_flag

    def set_checkpoint_processor(self, checkpoint_segments: int):
        """Checkpoint processor with configurable segments.

        Controls how the processor is checkpointed:
        - 0: Use processor's internal per-block checkpointing
        - -1: Checkpoint entire processor as one segment
        - N > 0: Checkpoint every N blocks (not yet implemented)

        Only effective when set_checkpoint_model(False).

        Parameters
        ----------
        checkpoint_segments : int
            Checkpointing strategy (0, -1, or positive integer).
        """
        self._checkpoint_processor_segments = checkpoint_segments

    def set_checkpoint_decoder(self, checkpoint_flag: bool):
        """Checkpoint decoder section.

        Checkpoints the decoder forward pass as a single segment.
        Only effective when set_checkpoint_model(False).

        Parameters
        ----------
        checkpoint_flag : bool
            If True, checkpoint decoder section.
        """
        self._checkpoint_decoder = checkpoint_flag

    def _set_layer_checkpointing(self, flag: bool):
        """Enable/disable fine-grained checkpointing in all layers."""
        # This would require modifying the encoder/processor/decoder
        # to support runtime checkpoint flag changes
        # For now, this is controlled via __init__ parameter
        pass

    def _encoder_forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encoder forward pass (for checkpointing)."""
        return self.encoder(features)

    def _processor_forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """Processor forward pass (for checkpointing)."""
        return self.processor(
            x,
            edge_index,
            edge_attr,
            batch_size=batch_size,
            efficient_batching=self.efficient_batching,
        )

    def _decoder_forward(
        self,
        processed_features: Tensor,
        original_features: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Decoder forward pass (for checkpointing)."""
        return self.decoder(processed_features, original_features, batch_size)

    def _custom_forward(self, features: Tensor) -> Tensor:
        """Forward pass with hierarchical checkpointing."""
        batch_size = features.shape[0]

        # Encoder
        if self._checkpoint_encoder:
            latent_features, edge_index, edge_attr = checkpoint(
                self._encoder_forward,
                features,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            latent_features, edge_index, edge_attr = self.encoder(features)

        # Processor
        if self._checkpoint_processor_segments == -1:
            # Checkpoint entire processor as one block
            processed_features = checkpoint(
                self._processor_forward,
                latent_features,
                edge_index,
                edge_attr,
                batch_size if self.efficient_batching else None,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            # Use processor's internal checkpointing (controlled by use_checkpointing)
            processed_features = self.processor(
                latent_features,
                edge_index,
                edge_attr,
                batch_size=batch_size,
                efficient_batching=self.efficient_batching,
            )

        # Decoder
        if self._checkpoint_decoder:
            output = checkpoint(
                self._decoder_forward,
                processed_features,
                features,
                batch_size,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            output = self.decoder(processed_features, features, batch_size)

        return output

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass through GraphCast model.

        Parameters
        ----------
        features : Tensor
            Input features of shape [batch_size, num_points, input_dim]

        Returns
        -------
        Tensor
            Output predictions of shape [batch_size, num_points, output_dim]
        """
        if self._checkpoint_model:
            # Checkpoint entire model as one segment
            return checkpoint(
                self._custom_forward,
                features,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            # Use hierarchical checkpointing
            return self._custom_forward(features)


class GraphCastConfig:
    """Configuration helper for GraphCast checkpointing strategies.

    Provides pre-defined checkpointing strategies for different use cases.
    """

    @staticmethod
    def no_checkpointing(model: GraphCast):
        """Disable all checkpointing (maximum speed, maximum memory)."""
        model.set_checkpoint_model(False)
        model.set_checkpoint_encoder(False)
        model.set_checkpoint_processor(0)
        model.set_checkpoint_decoder(False)

    @staticmethod
    def full_checkpointing(model: GraphCast):
        """Checkpoint entire model (maximum memory savings, slowest)."""
        model.set_checkpoint_model(True)

    @staticmethod
    def balanced_checkpointing(model: GraphCast):
        """Balanced strategy (good memory savings, moderate speed)."""
        model.set_checkpoint_model(False)
        model.set_checkpoint_encoder(True)
        model.set_checkpoint_processor(-1)
        model.set_checkpoint_decoder(True)

    @staticmethod
    def processor_only_checkpointing(model: GraphCast):
        """Checkpoint only processor (targets main memory bottleneck)."""
        model.set_checkpoint_model(False)
        model.set_checkpoint_encoder(False)
        model.set_checkpoint_processor(-1)
        model.set_checkpoint_decoder(False)
