
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData


class EnsembleGenerator(nn.Module):

    def __init__(self, noise_std: float = 0.1, method: str = "gaussian"):
        super().__init__()
        self.noise_std = noise_std
        self.method = method

    def forward(
        self,
        background_state: Union[torch.Tensor, Data, HeteroData],
        num_members: int,
    ) -> Union[torch.Tensor, Data, HeteroData]:
        
        if isinstance(background_state, (Data, HeteroData)):
            return self._graph_ensemble(background_state, num_members)
        return self._tensor_ensemble(background_state, num_members)

    def _tensor_ensemble(self, state: torch.Tensor, n: int) -> torch.Tensor:
        noise = torch.randn(
            state.shape[0], n, state.shape[1],
            device=state.device, dtype=state.dtype,
        ) * self.noise_std
        return state.unsqueeze(1).expand(-1, n, -1) + noise

    def _graph_ensemble(
        self, state: Union[Data, HeteroData], n: int
    ) -> Union[Data, HeteroData]:
        if isinstance(state, HeteroData):
            result = HeteroData()
            for nt in state.node_types:
                if hasattr(state[nt], "x") and state[nt].x is not None:
                    x = state[nt].x
                    if x.dim() == 2:
                        noise = torch.randn(
                            x.size(0), n, x.size(1), device=x.device, dtype=x.dtype
                        ) * self.noise_std
                        result[nt].x = x.unsqueeze(1).expand(-1, n, -1) + noise
                    else:
                        result[nt].x = x
                for k, v in state[nt].items():
                    if k != "x":
                        setattr(result[nt], k, v)
            for et in state.edge_types:
                for k, v in state[et].items():
                    setattr(result[et], k, v)
            return result

        result = Data()
        if hasattr(state, "x") and state.x is not None:
            x = state.x
            if x.dim() == 2:
                noise = torch.randn(
                    x.size(0), n, x.size(1), device=x.device, dtype=x.dtype
                ) * self.noise_std
                result.x = x.unsqueeze(1).expand(-1, n, -1) + noise
            else:
                result.x = state.x
        for k, v in state.items():
            if k != "x":
                setattr(result, k, v)
        return result


class KalmanFilterDA(nn.Module):
    

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

        self.ensemble_size = self.config.get("ensemble_size", 20)
        self.inflation_factor = self.config.get("inflation_factor", 1.1)
        self.observation_error_std = self.config.get("observation_error_std", 0.1)
        self.background_error_std = self.config.get("background_error_std", 0.5)

        self.ensemble_generator = EnsembleGenerator(
            noise_std=self.background_error_std, method="gaussian"
        )

        self.adaptive_inflation = self.config.get("adaptive_inflation", True)
        if self.adaptive_inflation:
            self.inflation_param = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        state_in: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
        ensemble_members: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Data, HeteroData]:
        
        if ensemble_members is None:
            ensemble = self.initialize_ensemble(state_in, self.ensemble_size)
        else:
            ensemble = ensemble_members

        updated_ensemble = self.assimilate(ensemble, observations)
        return self._compute_analysis(updated_ensemble)

    def initialize_ensemble(
        self, background_state: Union[torch.Tensor, Data, HeteroData], num_members: int
    ) -> Union[torch.Tensor, Data, HeteroData]:
        return self.ensemble_generator(background_state, num_members)

    def assimilate(
        self,
        ensemble: Union[torch.Tensor, Data, HeteroData],
        observations: torch.Tensor,
    ) -> Union[torch.Tensor, Data, HeteroData]:
  
        if isinstance(ensemble, torch.Tensor):
            return self._assimilate_tensor(ensemble, observations)
        elif isinstance(ensemble, (Data, HeteroData)):
            return self._assimilate_graph(ensemble, observations)
        else:
            raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")

    def _get_inflation(self) -> Union[float, torch.Tensor]:
        if self.adaptive_inflation:
            return 1.0 + torch.tanh(self.inflation_param)
        return self.inflation_factor

    def _assimilate_tensor(
        self, ensemble: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
       
        batch_size, num_members = ensemble.size(0), ensemble.size(1)
        orig_shape = ensemble.shape[2:]

        ensemble_mean = ensemble.mean(dim=1, keepdim=True)
        ensemble_perts = (ensemble - ensemble_mean) * self._get_inflation()
        inflated = ensemble_mean + ensemble_perts

        state_dim = int(torch.prod(torch.tensor(orig_shape)))
        obs_dim = observations.size(1)

        if state_dim >= obs_dim:
            ensemble_obs = inflated[:, :, :obs_dim]
        else:
            reps = obs_dim // state_dim + (1 if obs_dim % state_dim > 0 else 0)
            ensemble_obs = inflated[:, :, :state_dim].repeat(1, 1, reps)[:, :, :obs_dim]

        obs_mean = ensemble_obs.mean(dim=1, keepdim=True)
        obs_perts = ensemble_obs - obs_mean

        R = (self.observation_error_std ** 2) * torch.eye(
            obs_dim, device=observations.device, dtype=observations.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)

        P_hh = obs_perts.transpose(-2, -1) @ obs_perts / (num_members - 1)
        S = P_hh + R

        state_perts = inflated - inflated.mean(dim=1, keepdim=True)
        P_xh = state_perts.transpose(-2, -1) @ obs_perts / (num_members - 1)

        K = P_xh @ torch.inverse(S)
        innovation = observations.unsqueeze(1) - obs_mean
        dx = (K @ innovation.transpose(-2, -1)).squeeze(-1)

        dx_expanded = dx.unsqueeze(1).expand(-1, num_members, -1)
        return inflated + dx_expanded

    def _assimilate_graph(
        self, ensemble: Union[Data, HeteroData], observations: torch.Tensor
    ) -> Union[Data, HeteroData]:

        inflation = self._get_inflation()

        if isinstance(ensemble, HeteroData):
            result = HeteroData()
            for nt in ensemble.node_types:
                if hasattr(ensemble[nt], "x") and ensemble[nt].x is not None:
                    x = ensemble[nt].x
                    if x.dim() == 3:
                        x_t = x.transpose(1, 2)
                        mean = x_t.mean(dim=2, keepdim=True)
                        perts = (x_t - mean) * inflation
                        result[nt].x = (mean + perts).transpose(1, 2)
                    else:
                        result[nt].x = x
                else:
                    for k, v in ensemble[nt].items():
                        if k != "x":
                            setattr(result[nt], k, v)
            for et in ensemble.edge_types:
                for k, v in ensemble[et].items():
                    setattr(result[et], k, v)
            return result

        result = Data()
        if hasattr(ensemble, "x") and ensemble.x is not None:
            x = ensemble.x
            if x.dim() == 3:
                x_t = x.transpose(1, 2)
                mean = x_t.mean(dim=2, keepdim=True)
                perts = (x_t - mean) * inflation
                result.x = (mean + perts).transpose(1, 2)
            else:
                result.x = ensemble.x
        for k, v in ensemble.items():
            if k != "x":
                setattr(result, k, v)
        return result

    def _compute_analysis(
        self, ensemble: Union[torch.Tensor, Data, HeteroData]
    ) -> Union[torch.Tensor, Data, HeteroData]:

        if isinstance(ensemble, torch.Tensor):
            return ensemble.mean(dim=1)

        if isinstance(ensemble, HeteroData):
            result = HeteroData()
            for nt in ensemble.node_types:
                if hasattr(ensemble[nt], "x") and ensemble[nt].x is not None:
                    x = ensemble[nt].x
                    result[nt].x = x.mean(dim=1) if x.dim() == 3 else x
                else:
                    for k, v in ensemble[nt].items():
                        if k != "x":
                            setattr(result[nt], k, v)
            for et in ensemble.edge_types:
                for k, v in ensemble[et].items():
                    if (
                        k != "edge_attr"
                        or ensemble[et].edge_attr is None
                        or ensemble[et].edge_attr.dim() != 3
                    ):
                        setattr(result[et], k, v)
                    else:
                        ea = ensemble[et].edge_attr
                        result[et].edge_attr = ea.mean(dim=1) if ea.dim() == 3 else ea
            return result

        if isinstance(ensemble, Data):
            result = Data()
            if hasattr(ensemble, "x") and ensemble.x is not None:
                x = ensemble.x
                result.x = x.mean(dim=1) if x.dim() == 3 else x
            for k, v in ensemble.items():
                if k != "x":
                    setattr(result, k, v)
            return result

        raise TypeError(f"Unsupported ensemble type: {type(ensemble)}")
