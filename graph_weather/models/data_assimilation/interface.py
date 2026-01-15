"""
Data Assimilation Interface Module.

This module provides a unified interface for different data assimilation strategies
and handles the integration with various model types.
"""
from typing import Any, Dict, Literal, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

from .data_assimilation_base import DataAssimilationBase
from .kalman_filter_da import KalmanFilterDA
from .particle_filter_da import ParticleFilterDA
from .variational_da import VariationalDA


class DAInterface(nn.Module):
    """
    Unified interface for data assimilation strategies.
    
    This class provides a consistent API for different DA methods and handles
    the integration with various model architectures (graph-based, tensor-based, etc.).
    """
    
    def __init__(
        self,
        strategy: Literal['kalman', 'particle', 'variational'] = 'kalman',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the DA interface.
        
        Args:
            strategy: DA strategy to use ('kalman', 'particle', 'variational')
            config: Configuration dictionary for the chosen strategy
        """
        super().__init__()
        
        self.strategy = strategy
        self.config = config or {}
        
        # Initialize the appropriate DA module
        if strategy == 'kalman':
            self.da_module = KalmanFilterDA(self.config)
        elif strategy == 'particle':
            self.da_module = ParticleFilterDA(self.config)
        elif strategy == 'variational':
            self.da_module = VariationalDA(self.config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Perform data assimilation using the selected strategy.
        
        Args:
            state_in: Input state (graph or tensor)
            observations: Observation data
            ensemble_members: Optional pre-generated ensemble members
            
        Returns:
            Updated state in the same format as input
        """
        return self.da_module(state_in, observations, ensemble_members)
    
    def initialize_ensemble(
        self,
        background_state: Union[torch.Tensor, Data, HeteroData],
        num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Initialize ensemble members from background state.
        
        Args:
            background_state: Background state to generate ensemble from
            num_members: Number of ensemble members to generate
            
        Returns:
            Ensemble of states
        """
        return self.da_module.initialize_ensemble(background_state, num_members)
    
    def assimilate(
        self,
        ensemble: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:
        """
        Perform the assimilation step on ensemble members.
        
        Args:
            ensemble: Ensemble of states
            observations: Observation data
            
        Returns:
            Updated ensemble of states
        """
        return self.da_module.assimilate(ensemble, observations)
    
    def switch_strategy(
        self,
        new_strategy: Literal['kalman', 'particle', 'variational'],
        new_config: Optional[Dict[str, Any]] = None
    ):
        """
        Dynamically switch to a different DA strategy.
        
        Args:
            new_strategy: New DA strategy to use
            new_config: Configuration for the new strategy
        """
        config = new_config or self.config
        
        if new_strategy == self.strategy:
            return  # Already using this strategy
        
        if new_strategy == 'kalman':
            self.da_module = KalmanFilterDA(config)
        elif new_strategy == 'particle':
            self.da_module = ParticleFilterDA(config)
        elif new_strategy == 'variational':
            self.da_module = VariationalDA(config)
        else:
            raise ValueError(f"Unknown strategy: {new_strategy}")
        
        self.strategy = new_strategy
        self.config = config
    
    def get_strategy(self) -> str:
        """
        Get the current DA strategy.
        
        Returns:
            Current strategy name
        """
        return self.strategy


def create_da_module(
    strategy: Literal['kalman', 'particle', 'variational'] = 'kalman',
    config: Optional[Dict[str, Any]] = None
) -> DataAssimilationBase:
    """
    Factory function to create DA modules.
    
    Args:
        strategy: DA strategy to use
        config: Configuration for the DA module
        
    Returns:
        Initialized DA module
    """
    if strategy == 'kalman':
        return KalmanFilterDA(config)
    elif strategy == 'particle':
        return ParticleFilterDA(config)
    elif strategy == 'variational':
        return VariationalDA(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class ModelIntegratedDA(nn.Module):
    """
    Wrapper class to integrate DA with existing models.
    
    This class allows plugging DA functionality into existing weather models
    without modifying their internal architecture.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        da_interface: DAInterface,
        ensemble_size: int = 20,
        enable_da: bool = True
    ):
        """
        Initialize the integrated DA model.
        
        Args:
            base_model: The original weather prediction model
            da_interface: DA interface to use for assimilation
            ensemble_size: Size of ensemble to generate
            enable_da: Whether to enable DA (can be toggled on/off)
        """
        super().__init__()
        
        self.base_model = base_model
        self.da_interface = da_interface
        self.ensemble_size = ensemble_size
        self.enable_da = enable_da
    
    def forward(
        self,
        inputs: Union[torch.Tensor, Data, HeteroData],
        observations: Optional[torch.Tensor] = None,
        return_ensemble: bool = False
    ) -> Union[torch.Tensor, Data, HeteroData, Dict[str, Union[torch.Tensor, Data, HeteroData]]]:
        """
        Forward pass with optional data assimilation.
        
        Args:
            inputs: Input data for the base model
            observations: Observation data for DA (if None, only base model runs)
            return_ensemble: Whether to return ensemble predictions
            
        Returns:
            Model output, possibly with DA applied
        """
        # Get base model prediction
        base_prediction = self.base_model(inputs)
        
        if not self.enable_da or observations is None:
            if return_ensemble:
                # Generate ensemble from base prediction
                ensemble = self.da_interface.initialize_ensemble(
                    base_prediction, self.ensemble_size
                )
                return {
                    'prediction': base_prediction,
                    'ensemble': ensemble
                }
            else:
                return base_prediction
        
        # Perform ensemble generation and DA
        ensemble = self.da_interface.initialize_ensemble(base_prediction, self.ensemble_size)
        
        # Apply DA if observations are available
        updated_ensemble = self.da_interface.assimilate(ensemble, observations)
        
        # Compute analysis from updated ensemble
        analysis = self.da_interface._compute_analysis(updated_ensemble)
        
        if return_ensemble:
            return {
                'prediction': analysis,
                'ensemble': updated_ensemble,
                'base_prediction': base_prediction
            }
        else:
            return analysis
    
    def toggle_da(self, enable: bool):
        """
        Enable or disable data assimilation.
        
        Args:
            enable: Whether to enable DA
        """
        self.enable_da = enable
    
    def get_base_model(self) -> nn.Module:
        """
        Get the underlying base model.
        
        Returns:
            Base model
        """
        return self.base_model


def integrate_da_with_model(
    model: nn.Module,
    da_strategy: Literal['kalman', 'particle', 'variational'] = 'kalman',
    da_config: Optional[Dict[str, Any]] = None,
    ensemble_size: int = 20
) -> ModelIntegratedDA:
    """
    Integrate DA functionality with an existing model.
    
    Args:
        model: The original model to integrate with
        da_strategy: DA strategy to use
        da_config: Configuration for DA module
        ensemble_size: Size of ensemble to generate
        
    Returns:
        Integrated model with DA capability
    """
    da_interface = DAInterface(da_strategy, da_config)
    return ModelIntegratedDA(model, da_interface, ensemble_size)