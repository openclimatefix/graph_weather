from typing import List, Optional

import torch
import torch.nn as nn


class AIAssimilationNet(nn.Module):

    def __init__(
        self,
        state_size: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):

        super(AIAssimilationNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.state_size = state_size
        self.input_size = state_size * 2  # background + observations

        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build the network layers
        layers = []

        # Input layer
        prev_dim = self.input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer to produce analysis state
        layers.append(nn.Linear(prev_dim, state_size))

        self.network = nn.Sequential(*layers)

    def forward(self, first_guess: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        # Concatenate first-guess and observations
        combined_input = torch.cat([first_guess, observations], dim=-1)

        # Pass through the network to get the analysis
        analysis = self.network(combined_input)

        return analysis


class BlankFirstGuessGenerator(nn.Module):

    def __init__(self, state_size: int, init_value: float = 0.0):

        super(BlankFirstGuessGenerator, self).__init__()
        self.state_size = state_size
        self.init_value = init_value

    def forward(self, batch_size: int, device: torch.device = None) -> torch.Tensor:

        if device is None:
            device = torch.device("cpu")

        first_guess = torch.full(
            (batch_size, self.state_size), self.init_value, device=device, dtype=torch.float32
        )

        return first_guess
