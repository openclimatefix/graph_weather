"""
CaFA (Climate-Aware Factorized Attention)'s Architectural Design: 
- Transformer-based weather forecast for computational efficiency
- Uses Factorized Attention to reduce the cost of the attention mechanism
- A Three-Part System for Efficient Forecasting: Encoder, Factorized Transformer, Decoder
"""

from .encoder import CaFAEncoder
from .processor import CaFAProcessor
from .decoder import CaFADecoder
from .factorize import FactorizedAttention, FactorizedTransformerBlock, AxialAttention
from .model import CaFAForecaster

__version__ = "0.1.0"