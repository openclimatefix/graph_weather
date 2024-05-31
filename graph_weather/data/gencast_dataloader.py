"""
The dataloader for GenCast.

It has to:
- load, normalize and concatenate (across the channel dimension) the input timesteps 0 and 1.
- load and normalize the residual between timesteps 2 and 1.
- sample a noise level.
- corrupt the residual with noise generated at the given noise level.
"""

import einops
import numpy as np
import xarray as xr
from torch.utils.data import Dataset

from graph_weather.data import const
from graph_weather.models.gencast.utils.noise import generate_isotropic_noise, sample_noise_level


class GenCastDataset(Dataset):
    """
    Dataset class for GenCast training data.

    Args:
        obs_path: dataset path.
        atmospheric_features: list of features depending on pressure levels.
        single_features: list of features not depending on pressure levels.
        static_features: list of features not depending on time.
        max_year (optional): max year to include in training set. Defaults to 2018.
        time_step (optional): time step between predictions.
                    E.g. 12h steps correspond to time_step = 2 in a 6h dataset. Defaults to 2.
    """

    def __init__(
        self,
        obs_path,
        atmospheric_features,
        single_features,
        static_features,
        max_year=2018,
        time_step=2,
    ):
        """
        Initialize the GenCast dataset object.
        """
        super().__init__()
        self.data = xr.open_zarr(obs_path, chunks={})
        self.max_year = max_year

        self.num_lon = len(self.data["longitude"].values)
        self.num_lat = len(self.data["latitude"].values)
        self.num_vars = len(self.data.keys())
        self.pressure_levels = np.array(self.data["level"].values).astype(
            np.float32
        )  # Need them for loss weighting

        self.time_step = time_step  # e.g. 12h steps correspond to time_step = 2 in a 6h dataset

        self.atmospheric_features = atmospheric_features
        self.single_features = single_features
        self.static_features = static_features

        self.clock_features = ["local_time_of_the_day", "elapsed_year_progress"]
        # Lat and long will be added by the model itself in the graph

    def __len__(self):
        return sum(self.data["time.year"].values < self.max_year) - 2 * self.time_step

    def __getitem__(self, item):
        start = self.data.isel(time=[item, item + self.time_step])
        end = self.data.isel(time=item + 2 * self.time_step)

        # Stack atmospheric features for input
        atmospheric_input_data = np.stack(
            [
                (start[f"{var}"].values - np.array(const.ERA5_MEANS[f"{var}"])[None, :, None, None])
                / (np.array(const.ERA5_STD[f"{var}"])[None, :, None, None] + 0.0001)
                for var in self.atmospheric_features
            ],
            axis=-1,
        ).astype(np.float32)

        atmospheric_input_data = einops.rearrange(
            atmospheric_input_data, "t lev lon lat var -> t lon lat (var lev)"
        )
        atmospheric_input_data = np.nan_to_num(atmospheric_input_data)
        assert not np.isnan(atmospheric_input_data).any()

        # Stack single features for input
        single_input_data = np.stack(
            [
                (start[f"{var}"].values - np.array(const.ERA5_MEANS[f"{var}"]))
                / (np.array(const.ERA5_STD[f"{var}"]) + 0.0001)
                for var in self.single_features
            ],
            axis=-1,
        ).astype(np.float32)

        single_input_data = np.nan_to_num(single_input_data)
        assert not np.isnan(single_input_data).any()

        # Stack the static features for input
        static_input_data = np.stack(
            [
                (start[f"{var}"].values - np.array(const.ERA5_MEANS[f"{var}"]))
                / (np.array(const.ERA5_STD[f"{var}"]) + 0.0001)
                for var in self.static_features
            ],
            axis=-1,
        ).astype(np.float32)
        static_input_data = np.stack([static_input_data] * 2, axis=0)
        static_input_data = np.nan_to_num(static_input_data)
        assert not np.isnan(static_input_data).any()
        assert (static_input_data[0] == static_input_data[1]).all()

        # Stack the time features for input
        day_of_year = start.time.dt.dayofyear.values / 365.0
        sin_day_of_year = (
            np.sin(2 * np.pi * day_of_year)[:, None, None]
            * np.ones((self.num_lon, self.num_lat))[None, :, :]
        )
        cos_day_of_year = (
            np.cos(2 * np.pi * day_of_year)[:, None, None]
            * np.ones((self.num_lon, self.num_lat))[None, :, :]
        )

        local_mean_time = (
            np.ones((2, self.num_lon, self.num_lat)) * (start.time.dt.hour.values[:, None, None])
            + start["longitude"].values[None, :, None] * 4 / 60.0
        )
        sin_local_mean_time = np.sin(2 * np.pi * local_mean_time / 24.0)
        cos_local_mean_time = np.cos(2 * np.pi * local_mean_time / 24.0)

        clock_input_data = np.stack(
            [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], axis=-1
        )

        clock_input_data = np.nan_to_num(clock_input_data).astype(np.float32)
        assert not np.isnan(clock_input_data).any()

        # Stack atmospheric features for output
        atmospheric_output_data = np.stack(
            [
                (
                    (end[f"{var}"].values - start.isel(time=1)[f"{var}"].values)
                    - np.array(const.ERA5_DIFF_MEAN[f"{var}"])[:, None, None]
                )
                / (np.array(const.ERA5_DIFF_STD[f"{var}"])[:, None, None] + 0.0001)
                for var in self.atmospheric_features
            ],
            axis=-1,
        ).astype(np.float32)

        atmospheric_output_data = einops.rearrange(
            atmospheric_output_data, "lev lon lat var -> lon lat (var lev)"
        )
        atmospheric_output_data = np.nan_to_num(atmospheric_output_data)
        assert not np.isnan(atmospheric_output_data).any()

        # Stack single features for output
        single_output_data = np.stack(
            [
                (
                    (end[f"{var}"].values - start.isel(time=1)[f"{var}"].values)
                    - np.array(const.ERA5_DIFF_MEAN[f"{var}"])
                )
                / (np.array(const.ERA5_DIFF_STD[f"{var}"]) + 0.0001)
                for var in self.single_features
            ],
            axis=-1,
        ).astype(np.float32)

        single_output_data = np.nan_to_num(single_output_data)
        assert not np.isnan(single_output_data).any()

        inputs = np.concatenate(
            [atmospheric_input_data, single_input_data, static_input_data, clock_input_data],
            axis=-1,
        )
        inputs = np.concatenate([inputs[0], inputs[1]], axis=-1)

        target_residuals = np.concatenate([atmospheric_output_data, single_output_data], axis=-1)

        # Corrupt targets with noise
        noise_level = np.array([sample_noise_level()]).astype(np.float32)
        noise = generate_isotropic_noise(
            num_lat=self.num_lat, num_samples=target_residuals.shape[-1]
        )
        corrupted_residuals = target_residuals + noise_level * noise

        return (
            inputs,
            noise_level,
            corrupted_residuals,
            target_residuals,
        )


class BatchedGenCastDataset(Dataset):
    """
    Dataset class for GenCast batched training data.

    This dataset object returns a full batch as a single sample, it may be faster.

    Args:
        obs_path: Dataset path.
        atmospheric_features: List of features dependent on pressure levels.
        single_features: List of features not dependent on pressure levels.
        static_features: List of features not dependent on time.
        max_year (optional): Max year to include in training set. Defaults to 2018.
        time_step (optional): Time step between predictions.
                    E.g. 12h steps correspond to time_step = 2 in a 6h dataset. Defaults to 2.
        batch_size (optional): Size of the batch. Defaults to 32.
    """

    def __init__(
        self,
        obs_path,
        atmospheric_features,
        single_features,
        static_features,
        max_year=2018,
        time_step=2,
        batch_size=32,
    ):  
        """
        Initialize the GenCast dataset object.
        """
        super().__init__()
        self.data = xr.open_zarr(obs_path, chunks={})
        self.max_year = max_year

        self.num_lon = len(self.data["longitude"].values)
        self.num_lat = len(self.data["latitude"].values)
        self.num_vars = len(self.data.keys())
        self.pressure_levels = np.array(self.data["level"].values).astype(
            np.float32
        )  # Need them for loss weighting

        self.batch_size = batch_size
        self.time_step = time_step  # 12h steps correspond to time_step = 2 in a 6h dataset

        self.atmospheric_features = atmospheric_features
        self.single_features = single_features
        self.static_features = static_features

        self.clock_features = ["local_time_of_the_day", "elapsed_year_progress"]

        self.means, self.stds, self.diff_means, self.diff_stds = self._init_means_and_stds()
        # Lat and long will be added by the model itself in the graph

    def _init_means_and_stds(self):
        means = []
        stds = []
        diff_means = []
        diff_stds = []

        for var in self.atmospheric_features:
            means.extend(const.ERA5_MEANS[var])
            stds.extend(const.ERA5_STD[var])
            diff_means.extend(const.ERA5_DIFF_MEAN[var])
            diff_stds.extend(const.ERA5_DIFF_STD[var])

        for var in self.single_features:
            means.append(const.ERA5_MEANS[var])
            stds.append(const.ERA5_STD[var])
            diff_means.append(const.ERA5_DIFF_MEAN[var])
            diff_stds.append(const.ERA5_DIFF_STD[var])

        for var in self.static_features:
            means.append(const.ERA5_MEANS[var])
            stds.append(const.ERA5_STD[var])

        return (
            np.array(means).astype(np.float32),
            np.array(stds).astype(np.float32),
            np.array(diff_means).astype(np.float32),
            np.array(diff_stds).astype(np.float32),
        )

    def _normalize(self, data, means, stds):
        return (data - means) / (stds + 0.0001)

    def _batchify_inputs(self, data):
        start_idx = []
        for i in range(self.batch_size):
            start_idx.append([i, i + self.time_step])
        return data[start_idx]

    def _batchify_diffs(self, data):
        prev_idx = []
        target_idx = []
        for i in range(self.batch_size):
            prev_idx.append(i + self.time_step)
            target_idx.append(i + 2 * self.time_step)
        return data[target_idx] - data[prev_idx]

    def _generate_clock_features(self, ds):
        day_of_year = ds.time.dt.dayofyear.values / 365.0
        sin_day_of_year = (
            np.sin(2 * np.pi * day_of_year)[:, None, None]
            * np.ones((self.num_lon, self.num_lat))[None, :, :]
        )
        cos_day_of_year = (
            np.cos(2 * np.pi * day_of_year)[:, None, None]
            * np.ones((self.num_lon, self.num_lat))[None, :, :]
        )

        local_mean_time = (
            np.ones((self.num_lon, self.num_lat))[None, :, :]
            * (ds.time.dt.hour.values[:, None, None])
            + ds["longitude"].values[None, :, None] * 4 / 60.0
        )
        sin_local_mean_time = np.sin(2 * np.pi * local_mean_time / 24.0)
        cos_local_mean_time = np.cos(2 * np.pi * local_mean_time / 24.0)

        clock_input_data = np.stack(
            [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], axis=-1
        ).astype(np.float32)
        return clock_input_data

    def __len__(self):
        return sum(self.data["time.year"].values < self.max_year) - (
            3 * self.time_step + self.batch_size - 2
        )

    def __getitem__(self, item):
        # Compute the starting and ending point of the batch.
        starting_point = self.batch_size * item
        ending_point = starting_point + 3 * self.time_step + self.batch_size - 2

        # Load data
        ds = self.data.isel(time=np.arange(starting_point, ending_point))
        ds_atm = (
            ds[self.atmospheric_features]
            .to_array()
            .transpose("time", "longitude", "latitude", "level", "variable")
            .values
        )
        ds_atm = einops.rearrange(ds_atm, "t lon lat lev var -> t lon lat (var lev)")
        ds_single = (
            ds[self.single_features]
            .to_array()
            .transpose("time", "longitude", "latitude", "variable")
            .values
        )
        ds_static = (
            ds[self.static_features]
            .to_array()
            .transpose("longitude", "latitude", "variable")
            .values
        )
        ds_static = np.stack([ds_static] * (ending_point - starting_point), axis=0)

        # Compute inputs
        raw_inputs = np.concatenate([ds_atm, ds_single, ds_static], axis=-1)
        batched_inputs = self._batchify_inputs(raw_inputs)
        batched_inputs_norm = self._normalize(batched_inputs, self.means, self.stds)

        # Add time features
        ds_clock = self._batchify_inputs(self._generate_clock_features(ds))
        inputs = np.concatenate([batched_inputs_norm, ds_clock], axis=-1)
        # Concatenate timesteps
        inputs = np.concatenate([inputs[:, 0, :, :, :], inputs[:, 1, :, :, :]], axis=-1)
        inputs = np.nan_to_num(inputs).astype(np.float32)

        # Compute targets residuals
        raw_targets = np.concatenate([ds_atm, ds_single], axis=-1)
        batched_residuals = self._batchify_diffs(raw_targets)
        target_residuals = self._normalize(batched_residuals, self.diff_means, self.diff_stds)
        target_residuals = np.nan_to_num(target_residuals).astype(np.float32)

        # Corrupt targets with noise
        noise_levels = np.zeros((self.batch_size, 1), dtype=np.float32)
        corrupted_residuals = np.zeros_like(target_residuals, dtype=np.float32)
        for b in range(self.batch_size):
            noise_level = sample_noise_level()
            noise = generate_isotropic_noise(
                num_lat=self.num_lat, num_samples=target_residuals.shape[-1]
            )
            corrupted_residuals[b] = target_residuals[b] + noise_level * noise
            noise_levels[b] = noise_level

        return (inputs, noise_levels, corrupted_residuals, target_residuals)
