"""Aurora: A Foundation Model for Earth System Science.

- Combines 3D Swin Transformer encoding
- Perceiver processing for efficient computation
- 3D decoding for spatial-temporal predictions
"""

from .decoder import Decoder3D
from .encoder import Swin3DEncoder
from .model import AuroraModel, EarthSystemLoss
from .processor import PerceiverProcessor

__version__ = "0.1.0"

__all__ = [
    "AuroraModel",
    "EarthSystemLoss",
    "Swin3DEncoder",
    "Decoder3D",
    "PerceiverProcessor",
]

# Default configurations for different model sizes
MODEL_CONFIGS = {
    "tiny": {
        "in_channels": 1,
        "out_channels": 1,
        "embed_dim": 48,
        "latent_dim": 256,
        "spatial_shape": (16, 16, 16),
        "max_seq_len": 2048,
    },
    "base": {
        "in_channels": 1,
        "out_channels": 1,
        "embed_dim": 96,
        "latent_dim": 512,
        "spatial_shape": (32, 32, 32),
        "max_seq_len": 4096,
    },
    "large": {
        "in_channels": 1,
        "out_channels": 1,
        "embed_dim": 192,
        "latent_dim": 1024,
        "spatial_shape": (64, 64, 64),
        "max_seq_len": 8192,
    },
}


def create_model(config="base", **kwargs):
    """
    Create an Aurora model with specified configuration.

    Args:
        config (str): Model size configuration ('tiny', 'base', or 'large')
        **kwargs: Override default configuration parameters

    Returns:
        AuroraModel: Initialized model with specified configuration
    """
    if config not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown configuration: {config}. Choose from {list(MODEL_CONFIGS.keys())}"
        )

    # Start with default config and update with any provided kwargs
    model_config = MODEL_CONFIGS[config].copy()
    model_config.update(kwargs)

    return AuroraModel(**model_config)


def create_loss(alpha=0.5, beta=0.3, gamma=0.2):
    """
    Create an EarthSystemLoss instance with specified weights.

    Args:
        alpha (float): Weight for MSE loss
        beta (float): Weight for gradient loss
        gamma (float): Weight for physical consistency loss

    Returns:
        EarthSystemLoss: Initialized loss function
    """
    return EarthSystemLoss(alpha=alpha, beta=beta, gamma=gamma)
