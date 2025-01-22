import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class AnemoiDataset(Dataset):
    """
    Dataset for Anemoi weather datasets.

    Args:
        filepath: Path to the dataset in Zarr format.
        features: List of features to extract (e.g., "temperature", "geopotential").
        start_year: Initial year to filter the data. Defaults to 2016.
        end_year: Ending year to filter the data. Defaults to 2022.
        resolution: Resolution of the dataset (e.g., "5.625deg").
        land_sea_mask: Path to the land-sea mask dataset.
        orography: Path to the orography dataset.
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
    ):
        super().__init__()

        # Check time range validity
        assert (
            start_year <= end_year
        ), f"start_year ({start_year}) cannot be greater than end_year ({end_year})."

        # Load the main dataset
        self.data = xr.open_zarr(filepath)
        self.data = self.data.sel(time=slice(str(start_year), str(end_year)))

        # Load additional datasets
        if land_sea_mask:
            self.land_sea_mask = xr.open_zarr(land_sea_mask)
        if orography:
            self.orography = xr.open_zarr(orography)

        self.features = features
        self.resolution = resolution

        # Precompute geographical features
        self._compute_geographical_features()

        # Calculate mean and std for each feature
        self.calculate_feature_stats()

    def __len__(self):
        return len(self.data["time"])

    def __getitem__(self, idx):
        start = self.data.isel(time=idx)

        # Extract features
        input_data = self._extract_features(start)

        # Add precomputed geographical features
        input_data = np.concatenate((input_data, self.geo_features), axis=-1)

        # Rescale data using mean and std
        input_data = np.rescale_data(input_data)

        # Return tensor
        return ToTensor()(input_data)

    def _extract_features(self, data):
        """
        Extract the specified features and stack them into a single array.
        """
        feature_data = np.stack(
            [data[feature].values for feature in self.features], axis=-1
        ).astype(np.float32)

        return feature_data

    def _compute_geographical_features(self):
        """
        Compute geographical features: sin(lat), cos(lat), sin(lon), cos(lon), day-of-year.
        """
        lats = self.data["latitude"].values
        lons = self.data["longitude"].values

        sin_lat = np.sin(np.radians(lats))
        cos_lat = np.cos(np.radians(lats))
        sin_lon = np.sin(np.radians(lons))
        cos_lon = np.cos(np.radians(lons))

        # Add day-of-year feature
        times = self.data["time"].values
        days_of_year = (xr.DataArray(times).dt.dayofyear / 366.0).values

        self.geo_features = np.stack(
            [sin_lat, cos_lat, sin_lon, cos_lon, days_of_year], axis=-1
        ).astype(np.float32)

    def _calculate_feature_stats(self):
        """
        Calculate mean and std for each feature.
        """
        self.feature_mean = {}
        self.feature_std = {}

        for feature in self.features:
            feature_data = self.data[feature].values
            self.feature_mean[feature] = np.nanmean(feature_data)
            self.feature_std[feature] = np.nanstd(feature_data)

    def _rescale_data(self, data):
        """
        Rescale the data using the precomputed mean and std.
        """
        for i, feature in enumerate(self.features):
            mean = self.feature_mean.get(feature, 0)
            std = self.feature_std.get(feature, 1)
            data[..., i] = (data[..., i] - mean) / (std + 1e-6)

            return data
