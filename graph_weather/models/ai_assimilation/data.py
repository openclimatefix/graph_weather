

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class AIAssimilationDataset(Dataset):

    def __init__(
        self,
        first_guess_states: torch.Tensor,
        observations: torch.Tensor,
        observation_locations: Optional[torch.Tensor] = None,
    ):
        
        self.first_guess_states = first_guess_states
        self.observations = observations
        self.observation_locations = observation_locations

        # Validate dimensions
        msg = "First guess and observations must have same number of samples"
        assert first_guess_states.shape[0] == observations.shape[0], msg

    def __len__(self) -> int:
        return len(self.first_guess_states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        sample = {
            "first_guess": self.first_guess_states[idx],
            "observations": self.observations[idx],
        }

        if self.observation_locations is not None:
            sample["observation_locations"] = self.observation_locations[idx]

        return sample


def generate_synthetic_assimilation_data(
    num_samples: int = 1000,
    state_size: int = 100,
    obs_fraction: float = 0.5,
    bg_error_std: float = 0.5,
    obs_error_std: float = 0.3,
    spatial_correlation: bool = False,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Generate a true state with possible spatial correlation
    if spatial_correlation and grid_shape is not None:
        h, w = grid_shape
        if h * w != state_size:
            raise ValueError(f"Grid shape {grid_shape} doesn't match state size {state_size}")

        # Generate spatially correlated field using Gaussian random field
        true_state = torch.zeros(num_samples, state_size)

        for i in range(num_samples):
            # Create a 2D field with spatial correlation
            field_2d = torch.randn(h, w)

            # Apply simple smoothing to create spatial correlation
            for _ in range(3):  # Multiple smoothing iterations
                field_smooth = torch.zeros_like(field_2d)
                for row in range(h):
                    for col in range(w):
                        neighbors = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                neighbors.append(field_2d[nr, nc])

                        if neighbors:
                            field_smooth[row, col] = (field_2d[row, col] + sum(neighbors)) / (
                                1 + len(neighbors)
                            )
                        else:
                            field_smooth[row, col] = field_2d[row, col]

                field_2d = field_smooth

            true_state[i] = field_2d.flatten()
    else:
        # Generate uncorrelated random field
        true_state = torch.randn(num_samples, state_size)

    # Create first-guess states with errors relative to true state
    bg_errors = torch.randn(num_samples, state_size) * bg_error_std
    first_guess = true_state + bg_errors

    # Create observations with errors relative to true state
    obs_errors = torch.randn(num_samples, state_size) * obs_error_std
    observations = true_state + obs_errors

    # Apply observation fraction - zero out some observations
    obs_per_sample = int(state_size * obs_fraction)
    for i in range(num_samples):
        # Randomly select which observations to keep
        obs_indices = torch.randperm(state_size)[:obs_per_sample]
        mask = torch.zeros(state_size, dtype=torch.bool)
        mask[obs_indices] = True

        # Zero out non-observed values
        obs_masked = torch.zeros_like(observations[i])
        obs_masked[mask] = observations[i, mask]
        observations[i] = obs_masked

    return first_guess, observations, true_state


class AIAssimilationDataModule:

    def __init__(
        self,
        num_samples: int = 1000,
        state_size: int = 100,
        obs_fraction: float = 0.5,
        bg_error_std: float = 0.5,
        obs_error_std: float = 0.3,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        spatial_correlation: bool = False,
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
    
        self.num_samples = num_samples
        self.state_size = state_size
        self.obs_fraction = obs_fraction
        self.bg_error_std = bg_error_std
        self.obs_error_std = obs_error_std
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.spatial_correlation = spatial_correlation
        self.grid_shape = grid_shape

        # Will be populated by setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        # Generate synthetic data
        first_guess, observations, true_state = generate_synthetic_assimilation_data(
            num_samples=self.num_samples,
            state_size=self.state_size,
            obs_fraction=self.obs_fraction,
            bg_error_std=self.bg_error_std,
            obs_error_std=self.obs_error_std,
            spatial_correlation=self.spatial_correlation,
            grid_shape=self.grid_shape,
        )

        # Create the main dataset
        dataset = AIAssimilationDataset(first_guess, observations)

        # Calculate split sizes
        train_size = int(self.train_ratio * self.num_samples)
        val_size = int(self.val_ratio * self.num_samples)
        test_size = self.num_samples - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return self.test_loader


def create_observation_operator(
    state_size: int, obs_size: int, obs_locations: Optional[np.ndarray] = None
) -> torch.Tensor:
    
    if obs_locations is None:
        # Randomly select observation locations
        obs_indices = np.random.choice(state_size, size=obs_size, replace=False)
    else:
        obs_indices = obs_locations
        if len(obs_indices) != obs_size:
            raise ValueError(
                f"Number of obs_locations ({len(obs_indices)}) doesn't match obs_size ({obs_size})"
            )

    # Create H matrix as a selection matrix
    H = torch.zeros(obs_size, state_size)
    for i, idx in enumerate(obs_indices):
        if 0 <= idx < state_size:
            H[i, idx] = 1.0

    return H
