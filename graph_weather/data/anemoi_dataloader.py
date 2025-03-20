import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
try:
    from anemoi.datasets import open_dataset 
except ImportError as e:
    print(f"Import Error : {e}, use command `pip install anemoi-datasets`")
class AnemoiDataset(Dataset):
    """
    AnemoiDataset leverages the anemoi library to load, preprocess, and manage weather datasets.
    
    This implementation extends the basic functionality by supporting:
      - Spatial subsetting via latitude and longitude bounds.
      - Missing data handling (e.g., interpolation).
      - Computation of climatological statistics for bias correction or anomaly detection.
      - Ensemble and grid operations can be integrated further if needed.
    
    The dataset is expected to include:
      - A 'time' dimension and 1D spatial coordinates ('latitude' and 'longitude').
      - Meteorological features (e.g., "temperature", "geopotential", etc.) normalized using per-feature min–max scaling.
      - Additional geographical context computed from static features (sin/cos of lat/lon) and a dynamic day-of-year feature.
    
    Args:
        filepath (str): Path to the main dataset (in Zarr format) loaded via anemoi's open_dataset.
        features (list): List of meteorological features to extract.
        start_year (int): Starting year for filtering the data. Defaults to 2016.
        end_year (int): Ending year for filtering the data. Defaults to 2022.
        resolution (str): Dataset resolution (e.g., "5.625deg").
        land_sea_mask (str): Optional path to a land-sea mask dataset.
        orography (str): Optional path to an orography dataset.
        normalizer (callable): Optional callable to normalize data. If None, a default min–max scaler is applied.
        lat_bounds (tuple): Optional tuple (min_lat, max_lat) to subset spatially by latitude.
        lon_bounds (tuple): Optional tuple (min_lon, max_lon) to subset spatially by longitude.
        missing_method (str): Method to handle missing data. Default is 'interpolate'.
        compute_stats (bool): If True, compute climatological statistics for each feature.
    """
    def __init__(
        self,
        filepath: str,
        features: list,
        start_year: int = 2016,
        end_year: int = 2022,
        resolution: str = "5.625deg",
        land_sea_mask: str = None,
        orography: str = None,
        normalizer: callable = None,
        lat_bounds: tuple = None,
        lon_bounds: tuple = None,
        missing_method: str = 'interpolate',
        compute_stats: bool = False,
    ):
        super().__init__()
        assert start_year <= end_year, (
            f"start_year ({start_year}) cannot be greater than end_year ({end_year})."
        )

        # Load the main dataset using anemoi's open_dataset
        self.data = open_dataset(filepath)
        self.data = self.data.sel(time=slice(str(start_year), str(end_year)))
        
        # Optional: spatial subsetting based on provided latitude/longitude bounds.
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        if self.lat_bounds or self.lon_bounds:
            self._subset_data()

        # Optional: handle missing values
        self.missing_method = missing_method
        self._handle_missing_values()

        # Optionally load additional datasets
        if land_sea_mask:
            self.land_sea_mask = open_dataset(land_sea_mask)
        if orography:
            self.orography = open_dataset(orography)

        self.features = features
        self.resolution = resolution

        # Compute static geographical features (sin and cos transforms)
        self._compute_static_geographical_features()

        # Calculate per-feature min and max for normalization
        self.calculate_feature_stats()

        # Compute climatological statistics if required
        self.climatology = None
        if compute_stats:
            self.climatology = self.compute_climatology()

        # Set the normalizer
        self.normalizer = normalizer if normalizer is not None else self._default_normalizer

        # Transformer to convert numpy arrays to PyTorch tensors
        self.to_tensor = ToTensor()

    def _subset_data(self):
        """
        Subset the dataset spatially using provided latitude and longitude bounds.
        """
        if self.lat_bounds:
            self.data = self.data.sel(latitude=slice(self.lat_bounds[0], self.lat_bounds[1]))
        if self.lon_bounds:
            self.data = self.data.sel(longitude=slice(self.lon_bounds[0], self.lon_bounds[1]))

    def _handle_missing_values(self):
        """
        Handle missing values in the dataset using the specified method.
        Currently supports interpolation along spatial dimensions.
        """
        if self.missing_method == 'interpolate':
            # Use xarray's interpolation over spatial dims; time dimension left untouched.
            self.data = self.data.interpolate_na(dim="latitude", method="linear")
            self.data = self.data.interpolate_na(dim="longitude", method="linear")
        # Additional methods (e.g., masking, filling) can be implemented here.

    def compute_climatology(self):
        """
        Compute the climatological mean for each meteorological feature.
        
        Returns:
            dict: A dictionary with feature names as keys and corresponding climatological
                  means as numpy arrays.
        """
        climatology = {}
        for feature in self.features:
            # Compute the mean over the time dimension.
            climatology[feature] = self.data[feature].mean(dim="time").values.astype(np.float32)
        return climatology

    def __len__(self) -> int:
        """
        Return the number of time steps in the dataset.
        
        Returns:
            int: Number of samples based on the 'time' dimension.
        """
        return self.data.dims["time"]

    def __getitem__(self, idx: int):
        """
        Retrieve a processed sample from the dataset.
        
        Steps:
          1. Extract the time slice at the given index.
          2. Extract and stack meteorological features.
          3. Compute a normalized day-of-year feature.
          4. Concatenate meteorological and static geographical features.
          5. Apply normalization to meteorological channels.
          6. Convert to a PyTorch tensor.
        
        Returns:
            torch.Tensor: Tensor with shape (channels, height, width) containing processed data.
        """
        sample = self.data.isel(time=idx)
        meteo_data = self._extract_features(sample)

        # Compute day-of-year normalized (account for leap years)
        time_val = pd.Timestamp(sample["time"].values)
        doy_normalized = time_val.dayofyear / 366.0
        doy_feature = np.full(self.geo_static.shape[:2] + (1,), doy_normalized, dtype=np.float32)

        # Concatenate static geographical features with day-of-year feature
        geo_features = np.concatenate([self.geo_static, doy_feature], axis=-1)
        input_data = np.concatenate([meteo_data, geo_features], axis=-1)

        # Apply normalization on meteorological channels only
        input_data = self.normalizer(input_data)

        tensor_data = self.to_tensor(input_data)
        return tensor_data

    def _extract_features(self, sample: xr.Dataset) -> np.ndarray:
        """
        Extract and stack the specified meteorological features from a time slice.
        
        Args:
            sample (xr.Dataset): Single time slice from the dataset.
        
        Returns:
            np.ndarray: Array of shape (lat, lon, num_features) containing meteorological data.
        """
        feature_arrays = [
            sample[feature].values.astype(np.float32) for feature in self.features
        ]
        return np.stack(feature_arrays, axis=-1)

    def _compute_static_geographical_features(self):
        """
        Compute static geographical features from latitude and longitude:
          - sin(latitude), cos(latitude), sin(longitude), cos(longitude)
        
        The resulting features are stored in self.geo_static with shape (lat, lon, 4).
        Assumes the dataset contains 1D 'latitude' and 'longitude' coordinates.
        """
        lats = self.data["latitude"].values
        lons = self.data["longitude"].values
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        sin_lat = np.sin(np.radians(lat_grid))
        cos_lat = np.cos(np.radians(lat_grid))
        sin_lon = np.sin(np.radians(lon_grid))
        cos_lon = np.cos(np.radians(lon_grid))
        self.geo_static = np.stack([sin_lat, cos_lat, sin_lon, cos_lon], axis=-1).astype(np.float32)

    def calculate_feature_stats(self):
        """
        Calculate per-feature minimum and maximum values for normalization.
        Stores the statistics in self.feature_min and self.feature_max dictionaries.
        """
        self.feature_min = {}
        self.feature_max = {}
        for feature in self.features:
            vals = self.data[feature].values.astype(np.float32)
            self.feature_min[feature] = np.nanmin(vals)
            self.feature_max[feature] = np.nanmax(vals)

    def _default_normalizer(self, data: np.ndarray) -> np.ndarray:
        """
        Apply default min–max normalization to meteorological features only.
        The geographical features (last 5 channels) remain unchanged.
        
        Args:
            data (np.ndarray): Array of shape (H, W, C) where C = (num_features + 5).
        
        Returns:
            np.ndarray: Normalized data.
        """
        n_meteo = len(self.features)
        data_norm = data.copy()
        for i, feature in enumerate(self.features):
            min_val = self.feature_min[feature]
            max_val = self.feature_max[feature]
            range_val = max_val - min_val
            if range_val < 1e-6:
                data_norm[..., i] = 0.0
            else:
                data_norm[..., i] = (data_norm[..., i] - min_val) / range_val
        return data_norm
