"""
Data loader for self-supervised data assimilation framework
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AssimilationDataset(Dataset):
    """
    Dataset for self-supervised data assimilation
    Each sample contains background state and observations
    """

    def __init__(self, background_states, observations, true_states=None):
        """
        Initialize the assimilation dataset

        Args:
            background_states: Background states (x_b)
            observations: Observations (y)
            true_states: True states (for evaluation only, not used in training)
        """
        self.background_states = background_states
        self.observations = observations
        self.true_states = true_states

        assert len(background_states) == len(
            observations
        ), "Background and observation arrays must have same length"

        if true_states is not None:
            assert len(true_states) == len(
                background_states
            ), "True states must have same length as background states"

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.background_states)

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
        
        Returns:
            dict: Dictionary containing background, observations, and optionally true state.
        """
        bg = self.background_states[idx]
        obs = self.observations[idx]

        sample = {"background": bg, "observations": obs}

        if self.true_states is not None:
            sample["true_state"] = self.true_states[idx]

        return sample


def create_synthetic_assimilation_dataset(
    num_samples=1000,
    grid_size=(10, 10),
    num_channels=1,
    bg_error_std=0.5,
    obs_error_std=0.3,
    obs_fraction=0.5,
):
    """
    Create a synthetic dataset for data assimilation experiments

    Args:
        num_samples: Number of samples to generate
        grid_size: Size of spatial grid
        num_channels: Number of variables/channels
        bg_error_std: Standard deviation of background errors
        obs_error_std: Standard deviation of observation errors
        obs_fraction: Fraction of grid points that have observations

    Returns:
        dataset: AssimilationDataset object
    """
    total_size = np.prod(grid_size) if isinstance(grid_size, (tuple, list)) else grid_size

    # Generate true states with spatial correlation
    true_states = torch.randn(num_samples, num_channels, *grid_size)

    # Apply spatial smoothing to create realistic fields
    if len(grid_size) == 2:
        # Create a Gaussian smoothing kernel
        kernel_size = 5
        sigma = 1.0
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)

        # Apply smoothing to each sample and channel
        for i in range(num_samples):
            for c in range(num_channels):
                smoothed = torch.nn.functional.conv2d(
                    true_states[i : i + 1, c : c + 1], kernel, padding=kernel_size // 2, groups=1
                )
                true_states[i, c : c + 1] = smoothed

    # Create background states with errors
    bg_errors = torch.randn_like(true_states) * bg_error_std
    background_states = true_states + bg_errors

    # Create observations with errors
    obs_errors = torch.randn_like(true_states) * obs_error_std
    observations = true_states + obs_errors

    # Optionally mask some observations based on obs_fraction
    if obs_fraction < 1.0:
        mask = torch.rand_like(observations) < obs_fraction
        observations = observations * mask

    dataset = AssimilationDataset(background_states, observations, true_states)
    return dataset


def get_assimilation_data_loaders(
    dataset, batch_size=32, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True
):
    """
    Create train/validation/test data loaders from dataset

    Args:
        dataset: AssimilationDataset object
        batch_size: Size of batches
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        shuffle: Whether to shuffle the data

    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_observation_mask(grid_size, obs_fraction=0.5, seed=None):
    """
    Create a mask indicating which grid points have observations

    Args:
        grid_size: Size of the grid (can be int for 1D or tuple for 2D)
        obs_fraction: Fraction of grid points that have observations
        seed: Random seed for reproducibility

    Returns:
        mask: Boolean mask indicating observation locations
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    total_size = np.prod(grid_size) if isinstance(grid_size, (tuple, list)) else grid_size
    num_obs = int(total_size * obs_fraction)

    # Create random indices for observation locations
    obs_indices = np.random.choice(total_size, size=num_obs, replace=False)

    # Create mask
    mask_flat = torch.zeros(total_size, dtype=torch.bool)
    mask_flat[obs_indices] = True

    if isinstance(grid_size, (tuple, list)):
        mask = mask_flat.view(grid_size)
    else:
        mask = mask_flat

    return mask


def apply_observation_operator(data, obs_mask):
    """
    Apply observation operator to extract observed values from state

    Args:
        data: Full state data
        obs_mask: Boolean mask indicating observation locations

    Returns:
        observed_data: Data at observation locations only
    """
    if len(data.shape) > 2:  # Spatial data
        batch_size = data.size(0)
        reshaped_data = data.view(batch_size, -1)  # Flatten spatial dimensions
        observed_flat = reshaped_data * obs_mask.view(-1).float()
        return observed_flat.view_as(data)
    else:
        return data * obs_mask.float()


class AssimilationDataModule:
    """
    A PyTorch Lightning-style data module for assimilation data
    """

    def __init__(
        self,
        num_samples=1000,
        grid_size=(10, 10),
        num_channels=1,
        bg_error_std=0.5,
        obs_error_std=0.3,
        obs_fraction=0.5,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    ):
        """Initialize the data module.
            
        Args:
            num_samples: Number of samples to generate.
            grid_size: Size of spatial grid.
            num_channels: Number of variables/channels.
            bg_error_std: Standard deviation of background errors.
            obs_error_std: Standard deviation of observation errors.
            obs_fraction: Fraction of grid points that have observations.
            batch_size: Size of batches.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for testing.
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.bg_error_std = bg_error_std
        self.obs_error_std = obs_error_std
        self.obs_fraction = obs_fraction
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, stage=None):
        """Setup the dataset"""
        self.dataset = create_synthetic_assimilation_dataset(
            num_samples=self.num_samples,
            grid_size=self.grid_size,
            num_channels=self.num_channels,
            bg_error_std=self.bg_error_std,
            obs_error_std=self.obs_error_std,
            obs_fraction=self.obs_fraction,
        )

        self.train_loader, self.val_loader, self.test_loader = get_assimilation_data_loaders(
            self.dataset,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

    def train_dataloader(self):
        """Return the training data loader."""
        return self.train_loader

    def val_dataloader(self):
        """Return the validation data loader."""
        return self.val_loader

    def test_dataloader(self):
        """Return the test data loader."""
        return self.test_loader
