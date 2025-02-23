"""Statistics computation utils."""

import apache_beam  # noqa: F401
import numpy as np
import weatherbench2  # noqa: F401
import xarray as xr


def compute_statistics(dataset, vars, num_samples=100, single=False):
    """Compute statistics for single timestep.

    Args:
        dataset: xarray dataset.
        vars: list of features.
        num_samples (int, optional): _description_. Defaults to 100.
        single (bool, optional): if the features have multiple pressure levels. Defaults to False.

    Returns:
        means: dict with the means.
        stds: dict with the stds.
    """
    means = {}
    stds = {}
    for var in vars:
        print(f"Computing statistics for {var}")
        random_indexes = np.random.randint(0, len(dataset.time), num_samples)
        samples = data.isel(time=random_indexes)[var].values
        samples = np.nan_to_num(samples)
        axis_tuple = (0, 1, 2) if single else (0, 2, 3)
        means[var] = samples.mean(axis=axis_tuple)
        stds[var] = samples.std(axis=axis_tuple)
    return means, stds


def compute_statistics_diff(dataset, vars, num_samples=100, single=False, timestep=2):
    """Compute statistics for difference of two timesteps.

    Args:
        dataset: xarray dataset.
        vars: list of features.
        num_samples (int, optional): _description_. Defaults to 100.
        single (bool, optional): if the features have multiple pressure levels. Defaults to False.
        timestep (int, optional): number of steps to consider between start and end. Defaults to 2.

    Returns:
        means: dict with the means.
        stds: dict with the stds.
    """
    means = {}
    stds = {}
    for var in vars:
        print(f"Computing statistics for {var}")
        random_indexes = np.random.randint(0, len(dataset.time), num_samples)
        samples_start = data.isel(time=random_indexes)[var].values
        samples_start = np.nan_to_num(samples_start)
        samples_end = data.isel(time=random_indexes + timestep)[var].values
        samples_end = np.nan_to_num(samples_end)
        axis_tuple = (0, 1, 2) if single else (0, 2, 3)
        means[var] = (samples_end - samples_start).mean(axis=axis_tuple)
        stds[var] = (samples_end - samples_start).std(axis=axis_tuple)
    return means, stds


atmospheric_features = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

single_features = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    # "sea_surface_temperature",
    "total_precipitation_12hr",
]

static_features = ["geopotential_at_surface", "land_sea_mask"]

obs_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
# obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
# obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr'
data = xr.open_zarr(obs_path)
num_samples = 100
means, stds = compute_statistics_diff(data, single_features, num_samples=num_samples, single=True)
print("Means: ", means)
print("Stds: ", stds)
