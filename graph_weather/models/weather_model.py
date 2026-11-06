"""WeatherModel orchestrator for composing multiple stages of weather model components.

Supports chaining arbitrary sequences of encoder→processor→decoder stages,
where each stage's output becomes the next stage's input. All components
operate on PyTorch Geometric Data objects.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data

from graph_weather.models.protocols import StageConfig


class WeatherModel(torch.nn.Module):
    """Multi-stage weather model orchestrator.

    Composes multiple stages of encoder→processor→decoder components that
    implement the protocol-based interfaces. Each stage transforms a PyG
    Data object, and the output of one stage becomes the input to the next.

    Single-stage models are supported as a special case (one stage).
    Multi-stage models enable hierarchical encoding (e.g., coarse-to-fine
    or multi-level abstraction).

    Hierarchical checkpointing is supported via the set_checkpoint_* methods,
    following NVIDIA's pattern for memory-efficient training.
    """

    def __init__(
        self,
        stages: Union[List[StageConfig], StageConfig],
        checkpoint_model: bool = False,
    ):
        """
        Initialize the WeatherModel with one or more stages.

        Args:
            stages: A single StageConfig or a list of StageConfig objects.
                Each stage contains an encoder, processor, and decoder.
            checkpoint_model: If True, checkpoint the entire forward pass
                as a single segment (maximum memory savings).
        """
        super().__init__()
        self.checkpoint_model = checkpoint_model

        if isinstance(stages, StageConfig):
            stages = [stages]
        self.num_stages = len(stages)

        # Store components in lists for sequential access
        self.encoders = torch.nn.ModuleList([s.encoder for s in stages])
        self.processors = torch.nn.ModuleList([s.processor for s in stages])
        self.decoders = torch.nn.ModuleList([s.decoder for s in stages])

        # Checkpointing flags per stage
        self._checkpoint_encoder = [s.checkpoint_encoder for s in stages]
        self._checkpoint_processor = [s.checkpoint_processor for s in stages]
        self._checkpoint_decoder = [s.checkpoint_decoder for s in stages]

    def set_checkpoint_model(self, flag: bool):
        """Checkpoint the entire model as a single segment.

        Args:
            flag: If True, checkpoint the entire forward pass.
        """
        self.checkpoint_model = flag

    def set_checkpoint_stage(
        self, stage_idx: int, encoder: bool = False, processor: bool = False, decoder: bool = False
    ):
        """Set checkpointing flags for a specific stage.

        Args:
            stage_idx: Index of the stage (0-based).
            encoder: Checkpoint the encoder.
            processor: Checkpoint the processor.
            decoder: Checkpoint the decoder.
        """
        if 0 <= stage_idx < self.num_stages:
            self._checkpoint_encoder[stage_idx] = encoder
            self._checkpoint_processor[stage_idx] = processor
            self._checkpoint_decoder[stage_idx] = decoder

    def _stage_forward(
        self,
        stage_idx: int,
        data: Data,
        original_data: Optional[Data] = None,
    ) -> Tuple[Data, Data]:
        """Forward pass for a single stage (for checkpointing).

        Args:
            stage_idx: Index of the stage.
            data: Input PyG Data object.
            original_data: Original input data for the decoder (skip connection).

        Returns:
            Tuple of (output_data, stage_original_data) where stage_original_data
            is used for the next stage's decoder if chaining.
        """
        encoder = self.encoders[stage_idx]
        processor = self.processors[stage_idx]
        decoder = self.decoders[stage_idx]

        # Encoder
        if self._checkpoint_encoder[stage_idx]:
            latent = checkpoint(
                encoder.forward, data, use_reentrant=False, preserve_rng_state=False
            )
        else:
            latent = encoder(data)

        # Processor
        if self._checkpoint_processor[stage_idx]:
            processed = checkpoint(
                processor.forward, latent, use_reentrant=False, preserve_rng_state=False
            )
        else:
            processed = processor(latent)

        # Decoder
        stage_original = original_data if original_data is not None else data
        if self._checkpoint_decoder[stage_idx]:
            output = checkpoint(
                decoder.forward,
                processed,
                stage_original,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            output = decoder(processed, stage_original)

        return output, stage_original

    def _custom_forward(self, data: Data) -> Data:
        """Forward pass with hierarchical checkpointing across stages.

        Args:
            data: Input PyG Data object.

        Returns:
            Output PyG Data object after all stages.
        """
        # Keep track of the original data for the first stage's decoder
        original_data = data
        current_data = data

        for i in range(self.num_stages):
            current_data, _ = self._stage_forward(i, current_data, original_data)
            # For subsequent stages, the output of the previous stage becomes
            # the input, and we keep the original data for decoder skip connections
            if i < self.num_stages - 1:
                original_data = current_data

        return current_data

    def forward(self, data: Data) -> Data:
        """Forward pass through all stages.

        Args:
            data: Input PyG Data object with node features (`x`), edge index
                (`edge_index`), and optional edge attributes (`edge_attr`).

        Returns:
            Output PyG Data object after processing through all stages.
        """
        if self.checkpoint_model:
            return checkpoint(
                self._custom_forward, data, use_reentrant=False, preserve_rng_state=False
            )
        else:
            return self._custom_forward(data)


class WeatherModelConfig:
    """Configuration helper for WeatherModel checkpointing strategies."""

    @staticmethod
    def no_checkpointing(model: WeatherModel):
        """Disable all checkpointing (maximum speed, maximum memory)."""
        model.set_checkpoint_model(False)
        for i in range(model.num_stages):
            model.set_checkpoint_stage(i, encoder=False, processor=False, decoder=False)

    @staticmethod
    def full_checkpointing(model: WeatherModel):
        """Checkpoint entire model (maximum memory savings, slowest)."""
        model.set_checkpoint_model(True)

    @staticmethod
    def balanced_checkpointing(model: WeatherModel):
        """Balanced strategy (good memory savings, moderate speed)."""
        model.set_checkpoint_model(False)
        for i in range(model.num_stages):
            model.set_checkpoint_stage(i, encoder=True, processor=True, decoder=True)

    @staticmethod
    def processor_only_checkpointing(model: WeatherModel):
        """Checkpoint only processors (targets main memory bottleneck)."""
        model.set_checkpoint_model(False)
        for i in range(model.num_stages):
            model.set_checkpoint_stage(i, encoder=False, processor=True, decoder=False)
