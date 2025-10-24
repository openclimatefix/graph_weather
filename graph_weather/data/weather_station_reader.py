"""
The processor for weather station data.

It has to:
- scan directories for new observation files.
- convert raw CSV files into standardized NetCDF formats.
- execute multi-threaded processing to efficiently handle multiple files.
- integrate the processed data with meteorological models.
"""

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WeatherStationReader")

# Try importing synopticpy, but don't require it
# try:
#     from synoptic import Synoptic

#     SYNOPTIC_AVAILABLE = True
# except ImportError:
#     SYNOPTIC_AVAILABLE = False
#     logger.warning("SynopticPy package not installed, synoptic functionality won't be available")


class WeatherStationReader:
    """
    The reader for local weather station observations.

    It has to:
    - dynamically load new observation files from designated directories.
    - convert raw CSV files into standardized formats for weather model ingestion.
    - efficiently manage and retrieve processed observation data.
    - integrate seamlessly with meteorological model input requirements.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        file_pattern: str = "*.csv",
        max_workers: int = 4,
        cache_policy: str = "lru",
    ):
        """
        Initialize the weather station observation reader.

        Args:
            data_dir: Directory containing the raw observation files.
            cache_dir: Directory to store cached processed observations.
            file_pattern: Pattern to match observation files.
            max_workers: Number of parallel workers for file processing.
            cache_policy: Cache replacement policy ('lru', 'fifo')
        """
        # Initialize main directories and settings
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.file_pattern = file_pattern
        self.max_workers = max_workers
        self.processed_files = set()
        self._synopticpy_client = None

        # Configure cache settings
        self.cache_config = {"policy": cache_policy}

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def scan_for_new_observations(self) -> List[str]:
        """
        Scan for new observation files that haven't been processed yet.

        Returns:
            List of new files to be processed.
        """
        all_files = set(glob.glob(str(self.data_dir / self.file_pattern)))
        new_files = all_files - self.processed_files
        return list(new_files)

    def _process_file(self, filepath: str) -> Optional[str]:
        """
        Process a single observation file.

        Args:
            filepath: Path to the observation file to process.

        Returns:
            Path to the processed file or None if an error occurred.
        """
        try:
            # Read raw file (assuming CSV format)
            df = pd.read_csv(filepath)

            # Ensure we have the required columns for our tests
            required_columns = ["time", "station"]
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in {filepath}")
                    return None

            # Generate a cache filename
            filename = os.path.basename(filepath)
            cache_path = str(self.cache_dir / f"processed_{filename}.nc")

            # Convert time column to datetime if it's a string
            if df["time"].dtype == "object":
                df["time"] = pd.to_datetime(df["time"])

            # Create a proper multi-dimensional dataset
            # First, pivot the data to have time and station as dimensions
            # For each variable like temperature, pressure, etc.
            variables = [
                col
                for col in df.columns
                if col not in ["time", "station", "lat", "lon", "elevation"]
            ]

            # Create coordinates
            coords = {
                "time": sorted(df["time"].unique()),
                "station": sorted(df["station"].unique()),
            }

            # Create data variables
            data_vars = {}
            for var in variables:
                # Initialize with NaN
                var_data = np.full((len(coords["time"]), len(coords["station"])), np.nan)

                # Fill with actual data
                for i, row in df.iterrows():
                    time_idx = coords["time"].index(row["time"])
                    station_idx = coords["station"].index(row["station"])
                    var_data[time_idx, station_idx] = row[var]

                data_vars[var] = (["time", "station"], var_data)

            # Add station metadata as data variables
            for meta_var in ["lat", "lon", "elevation"]:
                if meta_var in df.columns:
                    # Get unique values per station
                    meta_values = []
                    for station in coords["station"]:
                        station_data = df[df["station"] == station][meta_var].iloc[0]
                        meta_values.append(station_data)

                    data_vars[meta_var] = (["station"], meta_values)

            # Create the dataset
            ds = xr.Dataset(data_vars=data_vars, coords=coords)

            # Save processed file
            ds.to_netcdf(cache_path)

            # Close the dataset to avoid file lock issues
            ds.close()

            return cache_path
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None

    def process_new_observations(self) -> List[str]:
        """
        Process all new observation files in parallel.

        Returns:
            List of paths to successfully processed files.
        """
        new_files = self.scan_for_new_observations()
        processed_paths = []

        if not new_files:
            return processed_paths

        logger.info(f"Processing {len(new_files)} new observation files")

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_file, new_files))

        for filepath, result in zip(new_files, results):
            if result:
                self.processed_files.add(filepath)
                processed_paths.append(result)

        logger.info(f"Successfully processed {len(processed_paths)} files")
        return processed_paths

    def get_observations_for_model(
        self,
        time_range: Optional[Tuple[str, str]] = None,
        region: Optional[Tuple[float, float, float, float]] = None,
        variables: Optional[List[str]] = None,
    ) -> Optional[xr.Dataset]:
        """
        Get processed observations ready for model ingestion.

        Args:
            time_range: Tuple of (start_time, end_time).
            region: Geographic region to filter observations (lon_min, lon_max, lat_min, lat_max).
            variables: List of variables to include.

        Returns:
            Combined dataset of processed observations or None if no data found.
        """
        # Process any new observations first
        self.process_new_observations()

        # List all processed files
        all_processed = list(self.cache_dir.glob("processed_*.nc"))

        if not all_processed:
            logger.warning("No processed files found")
            return None

        # Load and combine datasets
        datasets = []
        for file_path in all_processed:
            try:
                ds = xr.open_dataset(str(file_path))

                # Apply time filter
                if time_range and "time" in ds.dims:
                    start_time, end_time = time_range
                    ds = ds.sel(time=slice(start_time, end_time))

                # Apply spatial filtering
                if region and "lon" in ds and "lat" in ds:
                    lon_min, lon_max, lat_min, lat_max = region
                    ds = ds.where(
                        (ds.lon >= lon_min)
                        & (ds.lon <= lon_max)
                        & (ds.lat >= lat_min)
                        & (ds.lat <= lat_max),
                        drop=True,
                    )

                # Filter by variables
                if variables:
                    available_vars = list(ds.data_vars)
                    vars_to_keep = [v for v in variables if v in available_vars]
                    if vars_to_keep:
                        ds = ds[vars_to_keep]

                datasets.append(ds)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

        # Make sure to close datasets after use to prevent permission errors
        try:
            if datasets:
                # Combine datasets
                combined = xr.merge(datasets)

                # Close individual datasets
                for ds in datasets:
                    ds.close()

                return combined
            else:
                return None
        except Exception as e:
            logger.error(f"Error combining datasets: {str(e)}")
            # Make sure to close datasets in case of error
            for ds in datasets:
                ds.close()
            return None

    def _convert_to_synopticpy(self, observations: xr.Dataset) -> Optional[Dict]:
        """
        Convert observations to SynopticPy-compatible format.

        Args:
            observations: Dataset with observation data.

        Returns:
            Dictionary per SynopticPy specs, or None if empty
        """
        if observations is None:
            logger.warning("No observations to convert")
            return None

        # Create SynopticPy-compatible data structure
        synoptic_data = {"STATION": {}}

        # Process each station
        for station in observations.station.values:
            # Get station data
            station_data = observations.sel(station=station)

            # Extract coordinates if available
            lat = float(station_data.lat.values) if "lat" in station_data else None
            lon = float(station_data.lon.values) if "lon" in station_data else None
            elevation = (
                float(station_data.elevation.values) if "elevation" in station_data else None
            )

            # Initialize station entry
            synoptic_data["STATION"][str(station)] = {
                "NAME": f"Station {station}",
                "LATITUDE": lat,
                "LONGITUDE": lon,
                "ELEVATION": elevation,
                "OBSERVATIONS": {
                    "date_time": station_data.time.dt.strftime("%Y-%m-%dT%H:%M:%SZ").values.tolist()
                },
            }

            # Add variables
            for var_name in observations.data_vars:
                if (
                    var_name not in ["lat", "lon", "elevation"]
                    and "time" in observations[var_name].dims
                ):
                    # Extract values for this station and variable
                    values = station_data[var_name].values

                    # Add to observations
                    synoptic_data["STATION"][str(station)]["OBSERVATIONS"][
                        var_name
                    ] = values.tolist()

        return synoptic_data

    def convert_to_model_format(
        self, observations: Optional[xr.Dataset], model_format: str = "weatherreal"
    ) -> Optional[Union[xr.Dataset, Dict]]:
        """
        Convert observations to specific model format.

        Args:
            observations: Dataset containing observations.
            model_format: Target model format (weatherreal, synopticpy, etc.).

        Returns:
            Data in the format expected by the target model.
        """
        if observations is None:
            logger.warning("No observations to convert")
            return None

        # Use dictionary dispatch for cleaner code organization
        format_converters = {
            "weatherreal": self._convert_to_weatherreal,
            "synopticpy": self._convert_to_synopticpy,
        }

        converter = format_converters.get(model_format.lower())
        if converter:
            return converter(observations)
        else:
            logger.error(f"Unsupported model format: {model_format}")
            raise ValueError(f"Unsupported model format: {model_format}")

    def _convert_to_weatherreal(self, observations: xr.Dataset) -> xr.Dataset:
        """
        Convert observations to WeatherReal-Benchmark format.

        Args:
            observations: Dataset with observation data.

        Returns:
            Dataset formatted according to WeatherReal-Benchmark specifications.
        """
        # Create a new dataset with WeatherReal format
        weatherreal_data = observations.copy()

        # Ensure required dimensions exist
        required_dims = ["time", "station"]
        for dim in required_dims:
            if dim not in weatherreal_data.dims:
                raise ValueError(f"Required dimension '{dim}' not found in observations")

        # Rename variables to match WeatherReal conventions if needed
        var_mapping = {
            "temperature": "temperature",  # Example mapping
            "pressure": "pressure",
            "humidity": "humidity",
            "wind_speed": "wind_speed",
            # Add more mappings as needed
        }

        for old_name, new_name in var_mapping.items():
            if old_name in weatherreal_data and old_name != new_name:
                weatherreal_data = weatherreal_data.rename({old_name: new_name})

        # Add required attributes for WeatherReal compatibility
        weatherreal_data.attrs["source"] = "weather_station_reader"
        weatherreal_data.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add units if they don't exist
        default_units = {
            "temperature": "K",
            "pressure": "hPa",
            "humidity": "%",
            "wind_speed": "m/s",
        }

        for var_name, unit in default_units.items():
            if (
                var_name in weatherreal_data.data_vars
                and "units" not in weatherreal_data[var_name].attrs
            ):
                weatherreal_data[var_name].attrs["units"] = unit

        return weatherreal_data

    def convert_files_to_weatherreal(self, input_files: List[str], output_dir: str) -> List[str]:
        """
        Convert multiple observation files to WeatherReal format.

        Args:
            input_files: List of file paths to convert.
            output_dir: Directory to save the converted files.

        Returns:
            List of paths to the converted files.
        """
        os.makedirs(output_dir, exist_ok=True)
        converted_files = []

        for input_file in input_files:
            try:
                # Process the file
                if input_file.endswith(".csv"):
                    processed_path = self._process_file(input_file)
                    if processed_path is None:
                        continue

                    # Load the processed file
                    observations = xr.open_dataset(processed_path)
                elif input_file.endswith(".nc"):
                    # Directly load NetCDF files
                    observations = xr.open_dataset(input_file)
                else:
                    logger.warning(f"Unsupported file format: {input_file}")
                    continue

                # Convert to WeatherReal format
                weatherreal_data = self._convert_to_weatherreal(observations)

                # Create output filename
                base_name = os.path.basename(input_file)
                output_name = os.path.splitext(base_name)[0] + "_weatherreal.nc"
                output_path = os.path.join(output_dir, output_name)

                # Save in WeatherReal format
                weatherreal_data.to_netcdf(output_path)

                # Close datasets
                observations.close()
                weatherreal_data.close()

                converted_files.append(output_path)
                logger.info(f"Converted {input_file} to WeatherReal format at {output_path}")

            except Exception as e:
                logger.error(f"Error converting {input_file}: {str(e)}")

        return converted_files

    def initialize_synopticpy(
        self, token: Optional[str] = None, token_path: Optional[str] = None
    ) -> bool:
        """
        Initialize SynopticPy client for API-based data retrieval.

        Args:
            token: API token for SynopticPy.
            token_path: Path to file containing API token.

        Returns:
            True if initialization is successful, False otherwise.
        """
        try:
            if not SYNOPTIC_AVAILABLE:
                logger.warning(
                    "SynopticPy package is not installed - synoptic functionality unavailable"
                )
                return False

            # Get token from file if path is provided
            if token_path and not token:
                with open(token_path, "r") as f:
                    token = f.read().strip()

            if not token:
                logger.warning("No token provided for SynopticPy initialization")
                return False

            # Initialize SynopticPy client
            self._synopticpy_client = Synoptic(token)
            logger.info("SynopticPy client initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing SynopticPy: {str(e)}")
            return False

    def fetch_from_synopticpy(
        self,
        start_date: str,
        end_date: str,
        stids: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
    ) -> Optional[xr.Dataset]:
        """
        Fetch observations directly from SynopticPy API.

        Args:
            start_date: Start date (YYYY-MM-DD HH:MM).
            end_date: End date (YYYY-MM-DD HH:MM).
            stids: List of station IDs.
            variables: List of variables to fetch.

        Returns:
            Dataset with the observations or None if request failed.
        """
        if not self._synopticpy_client:
            if not self.initialize_synopticpy():
                logger.error("SynopticPy client not initialized")
                raise RuntimeError("SynopticPy client not initialized")

        try:
            # Use SynopticPy to fetch data
            stids_str = ",".join(stids) if stids else None
            variables_str = ",".join(variables) if variables else None

            data = self._synopticpy_client.get_observations(
                stids=stids_str, start=start_date, end=end_date, vars=variables_str
            )

            # Create xarray Dataset from the response
            if isinstance(data, dict) and "STATION" in data:
                stations = []
                times = []
                data_vars = {}

                # Extract all unique station IDs and times
                for station_id, station_info in data["STATION"].items():
                    stations.append(station_id)

                    if (
                        "OBSERVATIONS" in station_info
                        and "date_time" in station_info["OBSERVATIONS"]
                    ):
                        for dt in station_info["OBSERVATIONS"]["date_time"]:
                            if dt not in times:
                                times.append(dt)

                # Sort times
                times.sort()

                # Create coordinates
                coords = {"time": pd.DatetimeIndex(times), "station": stations}

                # Process variables and station metadata
                for station_id, station_info in data["STATION"].items():
                    # Get station metadata
                    lat = station_info.get("LATITUDE")
                    lon = station_info.get("LONGITUDE")
                    elevation = station_info.get("ELEVATION")

                    # Store metadata (only once)
                    if "lat" not in data_vars:
                        data_vars["lat"] = (["station"], [lat for _ in stations])
                    if "lon" not in data_vars:
                        data_vars["lon"] = (["station"], [lon for _ in stations])
                    if "elevation" not in data_vars:
                        data_vars["elevation"] = (["station"], [elevation for _ in stations])

                    # Process observation variables
                    if "OBSERVATIONS" in station_info:
                        obs = station_info["OBSERVATIONS"]

                        # For each variable except date_time
                        for var_name, values in obs.items():
                            if var_name != "date_time":
                                # Initialize if needed
                                if var_name not in data_vars:
                                    var_array = np.full((len(times), len(stations)), np.nan)
                                    data_vars[var_name] = (["time", "station"], var_array)

                                # Fill data
                                for i, dt in enumerate(obs["date_time"]):
                                    if i < len(values):  # Check bounds
                                        time_idx = times.index(dt)
                                        station_idx = stations.index(station_id)
                                        data_vars[var_name][1][time_idx, station_idx] = values[i]

                # Create the dataset
                ds = xr.Dataset(data_vars=data_vars, coords=coords)
                return ds
            else:
                logger.error("Unexpected response format from SynopticPy")
                raise ValueError("Unexpected response format from SynopticPy")
        except Exception as e:
            logger.error(f"Error fetching data from SynopticPy: {str(e)}")
            raise e

    def validate_observations(
        self, observations: xr.Dataset, qc_rules: Optional[Dict[str, Dict[str, float]]] = None
    ) -> xr.Dataset:
        """
        Apply quality control checks to observation data.

        Args:
            observations: Dataset with observations.
            qc_rules: Quality control rules to apply.

        Returns:
            Dataset with quality flags added.
        """
        # Default QC rules if none provided - based on typical meteorological thresholds
        # References: WMO Guidelines on Quality Control Procedures (WMO-No. 488)
        # and National Weather Service Observing Handbook No. 8
        if qc_rules is None:
            qc_rules = {
                "temperature": {"min": -80.0, "max": 60.0},  # °C, extreme Earth temperatures
                "pressure": {"min": 800.0, "max": 1100.0},  # hPa, standard range at sea level
                "humidity": {"min": 0.0, "max": 100.0},  # %, physical limits
                "wind_speed": {"min": 0.0, "max": 105.0},  # m/s, hurricane-force threshold
            }

        # Create quality flags
        for var_name, rules in qc_rules.items():
            if var_name in observations.data_vars:
                # Create mask for values outside acceptable range
                min_val = rules.get("min", float("-inf"))
                max_val = rules.get("max", float("inf"))

                # Create QC flag variable
                var_data = observations[var_name].values
                qc_mask = np.logical_or(
                    np.logical_or(var_data < min_val, var_data > max_val), np.isnan(var_data)
                )
                observations[f"{var_name}_qc"] = (observations[var_name].dims, qc_mask)

        return observations

    def interpolate_missing_data(
        self, observations: Optional[xr.Dataset], method: str = "linear"
    ) -> Optional[xr.Dataset]:
        """
        Interpolate missing data in time series.

        Args:
            observations: Dataset with observations.
            method: Interpolation method ('linear', 'nearest', etc.).

        Returns:
            Dataset with interpolated values.
        """
        if observations is None:
            return None

        # Create copy to avoid modifying original
        interpolated = observations.copy()

        # Interpolate along time dimension for each variable
        for var_name in observations.data_vars:
            if "time" in observations[var_name].dims and observations[var_name].isnull().any():
                interpolated[var_name] = observations[var_name].interpolate_na(
                    dim="time", method=method
                )

        return interpolated

    def resample_observations(
        self, observations: Optional[xr.Dataset], freq: str = "1H", aggregation: str = "mean"
    ) -> Optional[xr.Dataset]:
        """
        Resample observations to a different time frequency.

        Args:
            observations: Dataset with observations.
            freq: Target frequency ('1H', '1D', etc.).
            aggregation: Aggregation method ('mean', 'sum', 'min', 'max').

        Returns:
            Resampled dataset.
        """
        if observations is None or "time" not in observations.dims:
            return observations

        if aggregation == "mean":
            return observations.resample(time=freq).mean()
        elif aggregation == "sum":
            return observations.resample(time=freq).sum()
        elif aggregation == "min":
            return observations.resample(time=freq).min()
        elif aggregation == "max":
            return observations.resample(time=freq).max()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    def integrate_with_weatherreal(
        self, observations: Optional[xr.Dataset], output_path: str
    ) -> Optional[str]:
        """
        Save observations in WeatherReal-Benchmark compatible format.

        Args:
            observations: Dataset with observations.
            output_path: Path to save the output file.

        Returns:
            Path to the output file or None if an error occurred.
        """
        if observations is None:
            logger.warning("No observations to save")
            return None

        try:
            # Convert to WeatherReal format
            weatherreal_data = self._convert_to_weatherreal(observations)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # Save in NetCDF format
            weatherreal_data.to_netcdf(output_path)

            # Close the dataset to avoid file lock issues
            weatherreal_data.close()

            logger.info(f"Saved WeatherReal-compatible data to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving WeatherReal data: {str(e)}")
            return None

    def read_weatherreal_file(self, filepath: str) -> Optional[xr.Dataset]:
        """
        Read data directly from a WeatherReal-Benchmark formatted file.

        Args:
            filepath: Path to the WeatherReal-formatted NetCDF file.

        Returns:
            Dataset containing the loaded observations or None if an error occurred.
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None

            # Open the WeatherReal NetCDF file
            ds = xr.open_dataset(filepath)

            # Verify it has the expected structure
            required_dims = ["time", "station"]
            missing_dims = [dim for dim in required_dims if dim not in ds.dims]
            if missing_dims:
                logger.warning(f"Missing required dimensions {missing_dims} in WeatherReal file")
                # Still return the dataset even if dimensions are missing

            logger.info(f"Successfully loaded WeatherReal file: {filepath}")
            return ds

        except Exception as e:
            logger.error(f"Error reading WeatherReal file {filepath}: {str(e)}")
            return None
