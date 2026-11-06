from typing import Any, Dict, Literal, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

# Import with fallbacks to handle different execution contexts
try:
    # For relative import when used as part of package
    from .data_assimilation_base import DataAssimilationBase
    from .kalman_filter_da import KalmanFilterDA
    from .particle_filter_da import ParticleFilterDA
    from .variational_da import VariationalDA
except ImportError:
    try:
        # For absolute import when used as standalone
        from graph_weather.models.data_assimilation.data_assimilation_base import (
            DataAssimilationBase,
        )
        from graph_weather.models.data_assimilation.kalman_filter_da import KalmanFilterDA
        from graph_weather.models.data_assimilation.particle_filter_da import ParticleFilterDA
        from graph_weather.models.data_assimilation.variational_da import VariationalDA
    except ImportError:
        # For direct execution in isolated context
        import importlib.util
        import os

        # Load modules dynamically
        current_dir = os.path.dirname(__file__)

        # Load base module
        base_path = os.path.join(current_dir, "data_assimilation_base.py")
        spec = importlib.util.spec_from_file_location("data_assimilation_base", base_path)
        base_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_module)

        # Load Kalman module
        kalman_path = os.path.join(current_dir, "kalman_filter_da.py")
        spec = importlib.util.spec_from_file_location("kalman_filter_da", kalman_path)
        kalman_module = importlib.util.module_from_spec(spec)
        kalman_module.DataAssimilationBase = base_module.DataAssimilationBase
        kalman_module.EnsembleGenerator = base_module.EnsembleGenerator
        kalman_module.Data = __import__("torch_geometric.data").data.Data
        kalman_module.HeteroData = __import__("torch_geometric.data").data.HeteroData
        kalman_module.torch = __import__("torch")
        kalman_module.nn = __import__("torch.nn")
        kalman_module.typing = __import__("typing")
        spec.loader.exec_module(kalman_module)

        # Load Particle module
        particle_path = os.path.join(current_dir, "particle_filter_da.py")
        spec = importlib.util.spec_from_file_location("particle_filter_da", particle_path)
        particle_module = importlib.util.module_from_spec(spec)
        particle_module.DataAssimilationBase = base_module.DataAssimilationBase
        particle_module.EnsembleGenerator = base_module.EnsembleGenerator
        particle_module.Data = __import__("torch_geometric.data").data.Data
        particle_module.HeteroData = __import__("torch_geometric.data").data.HeteroData
        particle_module.torch = __import__("torch")
        particle_module.nn = __import__("torch.nn")
        particle_module.typing = __import__("typing")
        spec.loader.exec_module(particle_module)

        # Load Variational module
        var_path = os.path.join(current_dir, "variational_da.py")
        spec = importlib.util.spec_from_file_location("variational_da", var_path)
        var_module = importlib.util.module_from_spec(spec)
        var_module.DataAssimilationBase = base_module.DataAssimilationBase
        var_module.EnsembleGenerator = base_module.EnsembleGenerator
        var_module.Data = __import__("torch_geometric.data").data.Data
        var_module.HeteroData = __import__("torch_geometric.data").data.HeteroData
        var_module.torch = __import__("torch")
        var_module.nn = __import__("torch.nn")
        var_module.F = __import__("torch.nn.functional")
        var_module.typing = __import__("typing")
        spec.loader.exec_module(var_module)

        DataAssimilationBase = base_module.DataAssimilationBase
        KalmanFilterDA = kalman_module.KalmanFilterDA
        ParticleFilterDA = particle_module.ParticleFilterDA
        VariationalDA = var_module.VariationalDA


class DAInterface(nn.Module):

    def __init__(
        self,
        strategy: Literal["kalman", "particle", "variational"] = "kalman",
        config: Optional[Dict[str, Any]] = None,
    ):

        super().__init__()

        self.strategy = strategy
        self.config = config or {}

        # Initialize the appropriate DA module
        if strategy == "kalman":
            self.da_module = KalmanFilterDA(self.config)
        elif strategy == "particle":
            self.da_module = ParticleFilterDA(self.config)
        elif strategy == "variational":
            self.da_module = VariationalDA(self.config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Data, HeteroData]:

        return self.da_module(state_in, observations, ensemble_members)

    def initialize_ensemble(
        self, background_state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:

        return self.da_module.initialize_ensemble(background_state, num_members)

    def assimilate(
        self, ensemble: Union[torch.Tensor, Data, HeteroData], observations: torch.Tensor
    ) -> Union[torch.Tensor, Data, HeteroData]:

        return self.da_module.assimilate(ensemble, observations)

    def switch_strategy(
        self,
        new_strategy: Literal["kalman", "particle", "variational"],
        new_config: Optional[Dict[str, Any]] = None,
    ):

        config = new_config or self.config

        if new_strategy == self.strategy:
            return  # Already using this strategy

        if new_strategy == "kalman":
            self.da_module = KalmanFilterDA(config)
        elif new_strategy == "particle":
            self.da_module = ParticleFilterDA(config)
        elif new_strategy == "variational":
            self.da_module = VariationalDA(config)
        else:
            raise ValueError(f"Unknown strategy: {new_strategy}")

        self.strategy = new_strategy
        self.config = config

    def get_strategy(self) -> str:

        return self.strategy


def create_da_module(
    strategy: Literal["kalman", "particle", "variational"] = "kalman",
    config: Optional[Dict[str, Any]] = None,
) -> DataAssimilationBase:

    if strategy == "kalman":
        return KalmanFilterDA(config)
    elif strategy == "particle":
        return ParticleFilterDA(config)
    elif strategy == "variational":
        return VariationalDA(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class ModelIntegratedDA(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        da_interface: DAInterface,
        ensemble_size: int = 20,
        enable_da: bool = True,
    ):

        super().__init__()

        self.base_model = base_model
        self.da_interface = da_interface
        self.ensemble_size = ensemble_size
        self.enable_da = enable_da

    def forward(
        self,
        inputs: Union[torch.Tensor, Data, HeteroData],
        observations: Optional[torch.Tensor] = None,
        return_ensemble: bool = False,
    ) -> Union[torch.Tensor, Data, HeteroData, Dict[str, Union[torch.Tensor, Data, HeteroData]]]:

        # Get base model prediction
        base_prediction = self.base_model(inputs)

        if not self.enable_da or observations is None:
            if return_ensemble:
                # Generate ensemble from base prediction
                ensemble = self.da_interface.initialize_ensemble(
                    base_prediction, self.ensemble_size
                )
                return {"prediction": base_prediction, "ensemble": ensemble}
            else:
                return base_prediction

        # Perform ensemble generation and DA
        ensemble = self.da_interface.initialize_ensemble(base_prediction, self.ensemble_size)

        # Apply DA if observations are available
        updated_ensemble = self.da_interface.assimilate(ensemble, observations)

        # Compute analysis from updated ensemble
        analysis = self.da_interface.da_module._compute_analysis(updated_ensemble)

        if return_ensemble:
            return {
                "prediction": analysis,
                "ensemble": updated_ensemble,
                "base_prediction": base_prediction,
            }
        else:
            return analysis

    def toggle_da(self, enable: bool):
        self.enable_da = enable

    def get_base_model(self) -> nn.Module:
        return self.base_model


def integrate_da_with_model(
    model: nn.Module,
    da_strategy: Literal["kalman", "particle", "variational"] = "kalman",
    da_config: Optional[Dict[str, Any]] = None,
    ensemble_size: int = 20,
) -> ModelIntegratedDA:

    da_interface = DAInterface(da_strategy, da_config)
    return ModelIntegratedDA(model, da_interface, ensemble_size)
