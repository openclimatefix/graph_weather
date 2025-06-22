import numpy as np
import pandas as pd
import xarray as xr
from anemoi.datasets import open_dataset
import logging
from torch.utils.data import Dataset

class AnemoiDataset(Dataset):
    """
    Dataset class for Anemoi datasets integration with graph_weather.

    Args:
        dataset_name: Name of the Anemoi dataset (e.g., "era5-o48-2020-2021-6h-v1")
        features: List of atmospheric variables to use
        time_range: Optional tuple of (start_date, end_date)
        time_step: Time step between input and target (default: 1)
        max_samples: Maximum number of samples to use (for testing)
    """

    def __init__(
        self,
        dataset_name: str,
        features: list[str],
        time_range: tuple = None,
        time_step: int = 1,
        max_samples: int = None,
        **kwargs
    ):
        super().__init__()

        self.features = features
        self.time_step = time_step
        self.max_samples = max_samples
        
        # Build Anemoi dataset configuration
        config = {"dataset": dataset_name}
        if time_range:
            config["start"] = time_range[0]
            config["end"] = time_range[1]
        config.update(kwargs)
        
        # Load the dataset
        try:
            self.dataset = open_dataset(config)
            # Try different methods to get xarray data
            if hasattr(self.dataset, "to_xarray"):
                self.data = self.dataset.to_xarray()
            elif hasattr(self.dataset, "to_dataset"):
                self.data = self.dataset.to_dataset()
            else:
                self.data = self.dataset
            logging.info(f"Successfully loaded Anemoi dataset: {dataset_name}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Anemoi dataset '{dataset_name}': {e}. "
                "Please ensure the dataset is available and properly configured."
            )
        
        # Validate that we have the required features
        missing_features = [f for f in self.features if f not in self.data.data_vars]
        if missing_features:
            available_features = list(self.data.data_vars.keys())
            raise ValueError(
                f"Features {missing_features} not found in dataset. "
                f"Available features: {available_features}"
            )

        # Get grid information - try multiple coordinate name variations
        coord_names = ["latitude", "lat", "y"]
        self.grid_lat = None
        for name in coord_names:
            if name in self.data.coords:
                self.grid_lat = self.data.coords[name]
                break

        coord_names = ["longitude", "lon", "x"]
        self.grid_lon = None
        for name in coord_names:
            if name in self.data.coords:
                self.grid_lon = self.data.coords[name]
                break

        if self.grid_lat is None or self.grid_lon is None:
            available_coords = list(self.data.coords.keys())
            raise ValueError(f"Could not find latitude/longitude coordinates in dataset. "
                               f"Available coordinates: {available_coords}")
            
        self.num_lat = len(self.grid_lat)
        self.num_lon = len(self.grid_lon)

        # Initialize normalization parameters
        self.means, self.stds = self._init_normalization()
    
    def _init_normalization(self):
        """Initialize normalization parameters"""
        means = {}
        stds = {}

        for feature in self.features:
            if feature in self.data.data_vars:
                data_values = self.data[feature].values
                means[feature] = np.nanmean(data_values)
                stds[feature] = np.nanstd(data_values)
            else:
                logging.warning(f"Feature {feature} not found, using default normalization")
                means[feature] = 0.0
                stds[feature] = 1.0

        return means, stds

    def _normalize(self, data, feature):
        """Normalize data using feature-specific statistics"""
        return (data - self.means[feature]) / (self.stds[feature] + 1e-6)

    def _generate_clock_features(self, data_time):
        """Generate time features following GenCast pattern"""
        if hasattr(data_time, 'values'):
            timestamp = pd.Timestamp(data_time.values)
        else:
            timestamp = data_time
        
        day_of_year = timestamp.dayofyear / 365.0
        sin_day_of_year = np.sin(2 * np.pi * day_of_year)
        cos_day_of_year = np.cos(2 * np.pi * day_of_year)
        
        hour_of_day = timestamp.hour / 24.0
        sin_hour_of_day = np.sin(2 * np.pi * hour_of_day)
        cos_hour_of_day = np.cos(2 * np.pi * hour_of_day)
        
        num_locations = self.num_lat * self.num_lon
        time_features = np.column_stack(
            [
                np.full(num_locations, sin_day_of_year),
                np.full(num_locations, cos_day_of_year),
                np.full(num_locations, sin_hour_of_day),
                np.full(num_locations, cos_hour_of_day),
            ]
        )

        return time_features

    def __len__(self):
        total_length = len(self.data.time) - self.time_step
        if self.max_samples:
            return min(total_length, self.max_samples)
        return total_length

    def __getitem__(self, idx):
        input_data_slice = self.data.isel(time=idx)
        target_data_slice = self.data.isel(time=idx + self.time_step)
        
        input_features = []
        target_features = []

        for feature in self.features:
            if feature in self.data.data_vars:
                input_vals = input_data_slice[feature].values.reshape(-1)
                target_vals = target_data_slice[feature].values.reshape(-1)
                input_vals = self._normalize(input_vals, feature)
                target_vals = self._normalize(target_vals, feature)
                input_features.append(input_vals.reshape(-1, 1))
                target_features.append(target_vals.reshape(-1, 1))
        
        input_data = np.concatenate(input_features, axis=1)
        target_data = np.concatenate(target_features, axis=1)
        
        time_features = self._generate_clock_features(input_data_slice.time)
        input_data = np.concatenate([input_data, time_features], axis=1)
        target_data = np.concatenate([target_data, time_features], axis=1)

        return input_data.astype(np.float32), target_data.astype(np.float32)

    def get_dataset_info(self):
        """Return information about the loaded dataset"""
        return {
            "dataset_name": getattr(self, "dataset_name", "unknown"),
            "features": self.features,
            "grid_shape": (self.num_lat, self.num_lon),
            "time_steps": len(self.data.time),
            "dataset_length": len(self),
            "normalization_stats": {
                "means": self.means,
                "stds": self.stds
            }
        }
