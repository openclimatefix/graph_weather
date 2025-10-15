"""
CaFA (Climate-Aware Factorized Attention)'s Architectural Design:
- Transformer-based weather forecast for computational efficiency
- Uses Factorized Attention to reduce the cost of the attention mechanism
- A Three-Part System for Efficient Forecasting: Encoder, Factorized Transformer, Decoder
"""

from .decoder import CaFADecoder
from .encoder import CaFAEncoder
from .factorize import AxialAttention, FactorizedAttention, FactorizedTransformerBlock
from .model import CaFAForecaster
from .processor import CaFAProcessor

__version__ = "0.1.0"
