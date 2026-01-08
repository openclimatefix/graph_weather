"""
AI-based Data Assimilation Model

Implements a neural network that learns to produce analysis states by minimizing
the 3D-Var cost function without using ground-truth labels, as described in:
"AI-Based Data Assimilation: Learning the Functional of Analysis Estimation" (arXiv:2406.00390)

The model takes a first-guess state and observations as inputs and outputs an analysis state.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class AIAssimilationNet(nn.Module):
    """
    AI-based data assimilation network that learns to minimize the 3D-Var cost function.

    The network takes as input a first-guess (background) state and observations,
    and outputs an analysis state that optimally combines both according to the
    3D-Var cost function.
    """

    def __init__(
        self,
        state_size: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the AI-based assimilation network.

        Args:
            state_size: Size of the state vector (number of grid points or features)
            hidden_dims: List of hidden layer dimensions (default: [256, 256, 128])
            activation: Activation function ('relu', 'tanh', 'gelu')
            dropout_rate: Dropout rate for regularization
        """
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
        """
        Forward pass of the assimilation network.

        Args:
            first_guess: First-guess state (background state) [batch_size, state_size]
            observations: Observed values [batch_size, state_size]

        Returns:
            analysis: Analysis state that minimizes 3D-Var cost [batch_size, state_size]
        """
        # Concatenate first-guess and observations
        combined_input = torch.cat([first_guess, observations], dim=-1)

        # Pass through the network to get the analysis
        analysis = self.network(combined_input)

        return analysis


class SimpleAIBasedAssimilationNet(nn.Module):
    """
    Simplified version for spatial data using convolutional layers.

    This version is designed for grid-based meteorological data where spatial
    relationships are important.
    """

    def __init__(
        self,
        grid_size: tuple,
        num_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 3,
    ):
        """
        Initialize the simplified AI-based assimilation network for spatial data.

        Args:
            grid_size: Size of the spatial grid (height, width)
            num_channels: Number of channels/variables (e.g., temperature, pressure)
            hidden_channels: Number of hidden channels in convolutional layers
            num_layers: Number of convolutional layers
        """
        super(SimpleAIBasedAssimilationNet, self).__init__()

        self.grid_size = grid_size
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels

        # Input: 2 * num_channels (first_guess + observations) per spatial location
        input_channels = 2 * num_channels

        # Build convolutional layers
        layers = []

        # Initial layer
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, first_guess: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial data.

        Args:
            first_guess: First-guess state [batch_size, num_channels, height, width]
            observations: Observed values [batch_size, num_channels, height, width]

        Returns:
            analysis: Analysis state [batch_size, num_channels, height, width]
        """
        # Concatenate along the channel dimension
        combined_input = torch.cat(
            [first_guess, observations], dim=1
        )  # [batch_size, 2*num_channels, H, W]

        # Process through convolutional layers
        analysis = self.conv_layers(combined_input)

        return analysis


class BlankFirstGuessGenerator(nn.Module):
    """
    Generator for blank/zero first-guess states when no prior information is available.

    This can be used in cold-start scenarios where the first guess is initialized to zeros
    or a simple climatological mean.
    """

    def __init__(self, state_size: int, init_value: float = 0.0):
        """
        Initialize the blank first-guess generator.

        Args:
            state_size: Size of the state vector
            init_value: Initial value for the first guess (default: 0.0)
        """
        super(BlankFirstGuessGenerator, self).__init__()
        self.state_size = state_size
        self.init_value = init_value

    def forward(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate a blank first-guess state.

        Args:
            batch_size: Number of samples in the batch
            device: Device to create the tensor on

        Returns:
            first_guess: Blank first-guess state [batch_size, state_size]
        """
        if device is None:
            device = torch.device("cpu")

        first_guess = torch.full(
            (batch_size, self.state_size), self.init_value, device=device, dtype=torch.float32
        )

        return first_guess
