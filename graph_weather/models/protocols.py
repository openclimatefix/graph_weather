"""Protocol-based interfaces for weather model components.

Defines Python protocols (duck-typing) for Encoder, Processor, and Decoder
components that operate on PyTorch Geometric Data objects. Any component
implementing these protocols can be composed in a WeatherModel.
"""

from typing import Protocol

from torch_geometric.data import Data


class EncoderProtocol(Protocol):
    """Protocol for encoder components.

    An encoder transforms an input graph (node features, edge structure)
    into a latent representation graph. The input and output are both
    PyG Data objects.
    """

    def forward(self, data: Data) -> Data:
        """Encode input graph data to latent graph data.

        Args:
            data: Input PyG Data object with node features (`x`), edge index
                (`edge_index`), and optional edge attributes (`edge_attr`).

        Returns:
            PyG Data object representing the latent graph with encoded
            node features, edge index, and edge attributes.
        """
        ...


class ProcessorProtocol(Protocol):
    """Protocol for processor components.

    A processor transforms a latent graph by performing message passing
    or other graph operations. The input and output are both PyG Data
    objects.
    """

    def forward(self, data: Data) -> Data:
        """Process latent graph data.

        Args:
            data: Input PyG Data object (latent graph) with node features,
                edge index, and edge attributes.

        Returns:
            PyG Data object with processed node features. The edge structure
            may remain the same or be updated.
        """
        ...


class DecoderProtocol(Protocol):
    """Protocol for decoder components.

    A decoder transforms a latent graph back to the output graph domain.
    It takes the processed latent graph and the original input graph to
    produce the final output.
    """

    def forward(self, latent: Data, original: Data) -> Data:
        """Decode latent graph data back to output graph data.

        Args:
            latent: Processed latent graph from the processor.
            original: Original input graph (for skip connections or
                reference).

        Returns:
            PyG Data object representing the output graph with decoded
            node features.
        """
        ...


class StageConfig:
    """Configuration for a single stage (encoder + processor + decoder)."""

    def __init__(
        self,
        encoder: EncoderProtocol,
        processor: ProcessorProtocol,
        decoder: DecoderProtocol,
        checkpoint_encoder: bool = False,
        checkpoint_processor: bool = False,
        checkpoint_decoder: bool = False,
    ):
        """
        Initialize a stage configuration.

        Args:
            encoder: Component implementing EncoderProtocol.
            processor: Component implementing ProcessorProtocol.
            decoder: Component implementing DecoderProtocol.
            checkpoint_encoder: Whether to checkpoint the encoder.
            checkpoint_processor: Whether to checkpoint the processor.
            checkpoint_decoder: Whether to checkpoint the decoder.
        """
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.checkpoint_encoder = checkpoint_encoder
        self.checkpoint_processor = checkpoint_processor
        self.checkpoint_decoder = checkpoint_decoder
