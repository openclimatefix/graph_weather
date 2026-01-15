"""
Modular Data Assimilation Framework

This module provides a flexible, model-agnostic data assimilation framework
that can work with both graph-based and node-based representations.
"""
from .data_assimilation_base import DataAssimilationBase, EnsembleGenerator
from .interface import DAInterface
from .kalman_filter_da import KalmanFilterDA
from .particle_filter_da import ParticleFilterDA
from .variational_da import VariationalDA

__all__ = [
    "DataAssimilationBase",
    "EnsembleGenerator",
    "KalmanFilterDA",
    "ParticleFilterDA", 
    "VariationalDA",
    "DAInterface"
]