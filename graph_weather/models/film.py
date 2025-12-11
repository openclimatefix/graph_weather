# graph_weather/models/film.py
import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    One-hot -> MLP generator for FiLM params (MetNet style).
    Produces gamma, beta of shape (B, feature_dim).
    """

    def __init__(self, num_lead_times: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.num_lead_times = num_lead_times
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(num_lead_times, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )

    def forward(self, batch_size: int, lead_time: int, device=None):
        # one-hot: (B, num_lead_times)
        one_hot = torch.zeros(batch_size, self.num_lead_times, device=device)
        one_hot[:, lead_time] = 1.0
        gamma_beta = self.network(one_hot)  # (B, 2*feature_dim)
        gamma = gamma_beta[:, : self.feature_dim]
        beta = gamma_beta[:, self.feature_dim :]
        return gamma, beta


class FiLMApplier(nn.Module):
    """Apply FiLM: out = gamma * x + beta. Broadcasts to match x."""

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, N, C) or (B, C)
        # gamma/beta: (B, C)
        # Expand gamma/beta until dims match
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return x * gamma + beta
