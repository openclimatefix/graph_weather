import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma and beta) from a lead-time index.

    A one-hot vector for the given lead time is expanded to the batch size
    and passed through a small MLP to produce FiLM modulation parameters.

    Args:
        num_lead_times (int): Number of possible lead-time categories.
        hidden_dim (int): Hidden size for the internal MLP.
        feature_dim (int): Output dimensionality of gamma and beta.
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
        """
        Compute FiLM gamma and beta parameters.

        Args:
            batch_size (int): Number of samples to generate parameters for.
            lead_time (int): Lead-time index used to construct the one-hot input.
            device (optional): Device to place tensors on. Defaults to CPU.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                gamma: Tensor of shape (batch_size, feature_dim).
                beta:  Tensor of shape (batch_size, feature_dim).
        """

        one_hot = torch.zeros(batch_size, self.num_lead_times, device=device)
        one_hot[:, lead_time] = 1.0
        gamma_beta = self.network(one_hot)
        gamma = gamma_beta[:, : self.feature_dim]
        beta = gamma_beta[:, self.feature_dim :]
        return gamma, beta


class FiLMApplier(nn.Module):
    """
    Applies FiLM modulation to an input tensor.

    Gamma and beta are broadcast to match the dimensionality of the input,
    and the FiLM operation is applied elementwise.
    """

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, ...).
            gamma (torch.Tensor): Scaling parameters of shape (B, C).
            beta (torch.Tensor): Bias parameters of shape (B, C).

        Returns:
            torch.Tensor: Output tensor after FiLM modulation, same shape as `x`.
        """

        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return x * gamma + beta
