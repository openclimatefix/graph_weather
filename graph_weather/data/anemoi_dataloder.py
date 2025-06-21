import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from anemoi.datasets import open_dataset

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
        
        # Set features FIRST to avoid AttributeError
        self.features = features
        self.time_step = time_step
        self.max_samples = max_samples
        
        # Build Anemoi dataset configuration
        config = {"dataset": dataset_name}
        if time_range:
            config["start"] = time_range[0]  # Fix: use [0] for start
            config["end"] = time_range[1]    # Fix: use [1] for end
        config.update(kwargs)
        
        # Load the dataset
        try:
            self.dataset = open_dataset(config)
            # Try different methods to get xarray data
            if hasattr(self.dataset, 'to_xarray'):
                self.data = self.dataset.to_xarray()
            elif hasattr(self.dataset, 'to_dataset'):
                self.data = self.dataset.to_dataset()
            else:
                # Assume it's already xarray-like
                self.data = self.dataset
        except Exception as e:
            print(f"Warning: Could not load Anemoi dataset: {e}")
            # Create mock data for development/testing
            self.data = self._create_mock_data()
        
        # Get grid information - try multiple coordinate name variations
        coord_names = ['latitude', 'lat', 'y']
        self.grid_lat = None
        for name in coord_names:
            if name in self.data.coords:
                self.grid_lat = self.data.coords[name]
                break
        
        coord_names = ['longitude', 'lon', 'x']
        self.grid_lon = None
        for name in coord_names:
            if name in self.data.coords:
                self.grid_lon = self.data.coords[name]
                break
        
        if self.grid_lat is None or self.grid_lon is None:
            raise ValueError("Could not find latitude/longitude coordinates in dataset")
            
        self.num_lat = len(self.grid_lat)
        self.num_lon = len(self.grid_lon)
        
        # Initialize normalization parameters
        self.means, self.stds = self._init_normalization()
    
    def _create_mock_data(self):
        """Create mock data for testing when real datasets aren't available"""
        print("Creating mock Anemoi data for testing...")
        
        # Create mock coordinates
        lat = np.linspace(-90, 90, 32)
        lon = np.linspace(0, 360, 64)
        time = pd.date_range('2020-01-01', periods=100, freq='6h')  # Fix: use 'h' instead of 'H'
        
        # Create mock data
        data_vars = {}
        for feature in self.features:
            # Generate realistic-looking random data
            shape = (len(time), len(lat), len(lon))
            data_vars[feature] = (['time', 'latitude', 'longitude'], 
                                np.random.randn(*shape) * 10 + 273.15)  # Temperature-like
        
        return xr.Dataset(
            data_vars,
            coords={'time': time, 'latitude': lat, 'longitude': lon}
        )
    
    def _init_normalization(self):
        """Initialize normalization parameters"""
        means = {}
        stds = {}
        
        for feature in self.features:
            if feature in self.data.data_vars:
                # Calculate statistics from the data
                data_values = self.data[feature].values
                means[feature] = np.nanmean(data_values)
                stds[feature] = np.nanstd(data_values)
            else:
                # Use default values if feature not found
                print(f"Warning: Feature {feature} not found, using default normalization")
                means[feature] = 0.0
                stds[feature] = 1.0
        
        return means, stds
    
    def _normalize(self, data, feature):
        """Normalize data using feature-specific statistics"""
        return (data - self.means[feature]) / (self.stds[feature] + 1e-6)
    
    def _generate_clock_features(self, data_time):
        """Generate time features following GenCast pattern"""
        # Convert xarray DataArray to pandas Timestamp if needed
        if hasattr(data_time, 'values'):
            timestamp = pd.Timestamp(data_time.values)
        else:
            timestamp = data_time
        
        # Day of year embedding
        day_of_year = timestamp.dayofyear / 365.0
        sin_day_of_year = np.sin(2 * np.pi * day_of_year)
        cos_day_of_year = np.cos(2 * np.pi * day_of_year)
        
        # Hour of day embedding  
        hour_of_day = timestamp.hour / 24.0
        sin_hour_of_day = np.sin(2 * np.pi * hour_of_day)
        cos_hour_of_day = np.cos(2 * np.pi * hour_of_day)
        
        # Broadcast to all grid points
        num_locations = self.num_lat * self.num_lon
        time_features = np.column_stack([
            np.full(num_locations, sin_day_of_year),
            np.full(num_locations, cos_day_of_year),
            np.full(num_locations, sin_hour_of_day),
            np.full(num_locations, cos_hour_of_day)
        ])
        
        return time_features
    
    def __len__(self):
        total_length = len(self.data.time) - self.time_step
        if self.max_samples:
            return min(total_length, self.max_samples)
        return total_length
    
    def __getitem__(self, idx):
        # Get input and target time steps
        input_data_slice = self.data.isel(time=idx)
        target_data_slice = self.data.isel(time=idx + self.time_step)
        
        # Extract and normalize features
        input_features = []
        target_features = []
        
        for feature in self.features:
            if feature in self.data.data_vars:
                # Get data and reshape to (num_locations, 1) - flatten spatial dimensions
                input_vals = input_data_slice[feature].values.reshape(-1)  # Flatten all spatial dims
                target_vals = target_data_slice[feature].values.reshape(-1)
                
                # Normalize
                input_vals = self._normalize(input_vals, feature)
                target_vals = self._normalize(target_vals, feature)
                
                input_features.append(input_vals.reshape(-1, 1))
                target_features.append(target_vals.reshape(-1, 1))
        
        # Concatenate all features
        input_data = np.concatenate(input_features, axis=1)
        target_data = np.concatenate(target_features, axis=1)
        
        # Add time features (following GenCast pattern)
        time_features = self._generate_clock_features(input_data_slice.time)
        input_data = np.concatenate([input_data, time_features], axis=1)
        target_data = np.concatenate([target_data, time_features], axis=1)
        
        return input_data.astype(np.float32), target_data.astype(np.float32)
