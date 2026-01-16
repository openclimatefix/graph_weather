"""
OpenMeteo weather data fetcher for grid-based weather data.

It has to:
- fetch weather data from OpenMeteo API for forecast and historical endpoints.
- return properly structured xarray Datasets with gridded lat/lon coordinates.
- handle rate limiting to respect OpenMeteo free tier limits.
- cache fetched data to avoid redundant API calls.
- support parallel fetching for efficient grid data retrieval.
"""

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OpenMeteoWeatherDataFetcher")

# API endpoints
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

# Default hourly parameters to fetch
DEFAULT_HOURLY_PARAMETERS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]

# Available NWP models for forecast
FORECAST_NWP_MODELS = [
    "best_match",
    "gfs_seamless",
    "gfs_global",
    "ecmwf_ifs04",
    "icon_seamless",
    "icon_global",
    "gem_seamless",
    "gem_global",
    "meteofrance_seamless",
    "jma_seamless",
]

# Available NWP models for historical data
HISTORICAL_NWP_MODELS = [
    "era5",
    "era5_land",
    "cerra",
    "best_match",
]


class OpenMeteoError(Exception):
    """Base exception for OpenMeteo fetcher."""

    pass


class OpenMeteoRateLimitError(OpenMeteoError):
    """Raised when API rate limit is exceeded."""

    pass


class OpenMeteoAPIError(OpenMeteoError):
    """Raised when API returns an error."""

    pass


class OpenMeteoDataError(OpenMeteoError):
    """Raised when data assembly fails."""

    pass


class RateLimiter:
    """
    Token bucket rate limiter for OpenMeteo API calls.

    It has to:
    - enforce OpenMeteo free tier rate limits (600/min, 5000/hour, 10000/day, 300000/month).
    - track token consumption across multiple time windows.
    - block requests when limits are reached until tokens are refilled.
    - provide thread-safe rate limiting for parallel requests.
    """

    def __init__(
        self,
        requests_per_minute: int = 600,
        requests_per_hour: int = 5000,
        requests_per_day: int = 10000,
        requests_per_month: int = 300000,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute (default: 600).
            requests_per_hour: Maximum requests allowed per hour (default: 5000).
            requests_per_day: Maximum requests allowed per day (default: 10000).
            requests_per_month: Maximum requests allowed per month (default: 300000).
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.requests_per_month = requests_per_month

        self.minute_tokens = requests_per_minute
        self.hour_tokens = requests_per_hour
        self.day_tokens = requests_per_day
        self.month_tokens = requests_per_month

        self.last_minute_refill = time.time()
        self.last_hour_refill = time.time()
        self.last_day_refill = time.time()
        self.last_month_refill = time.time()

        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            current_time = time.time()

            # Refill minute bucket
            elapsed_minute = current_time - self.last_minute_refill
            if elapsed_minute >= 60:
                self.minute_tokens = self.requests_per_minute
                self.last_minute_refill = current_time
            else:
                refill = int(elapsed_minute * self.requests_per_minute / 60)
                self.minute_tokens = min(self.requests_per_minute, self.minute_tokens + refill)
                if refill > 0:
                    self.last_minute_refill = current_time

            # Refill hour bucket
            elapsed_hour = current_time - self.last_hour_refill
            if elapsed_hour >= 3600:
                self.hour_tokens = self.requests_per_hour
                self.last_hour_refill = current_time

            # Refill day bucket
            elapsed_day = current_time - self.last_day_refill
            if elapsed_day >= 86400:
                self.day_tokens = self.requests_per_day
                self.last_day_refill = current_time

            # Refill month bucket (approximate 30 days)
            elapsed_month = current_time - self.last_month_refill
            if elapsed_month >= 2592000:  # 30 days in seconds
                self.month_tokens = self.requests_per_month
                self.last_month_refill = current_time

            # Wait if any bucket is exhausted
            self._wait_for_tokens()

            # Consume token from all buckets
            self.minute_tokens -= 1
            self.hour_tokens -= 1
            self.day_tokens -= 1
            self.month_tokens -= 1

    def _wait_for_tokens(self) -> None:
        """Wait until tokens are available in all buckets."""
        # Check minute limit
        if self.minute_tokens <= 0:
            sleep_time = 60 - (time.time() - self.last_minute_refill)
            if sleep_time > 0:
                logger.info(f"Minute rate limit reached (600/min). Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            self.minute_tokens = self.requests_per_minute
            self.last_minute_refill = time.time()

        # Check hour limit
        if self.hour_tokens <= 0:
            sleep_time = 3600 - (time.time() - self.last_hour_refill)
            if sleep_time > 0:
                logger.warning(
                    f"Hourly rate limit reached (5000/hour). Waiting {sleep_time:.1f}s "
                    f"({sleep_time / 60:.1f} minutes)..."
                )
                time.sleep(sleep_time)
            self.hour_tokens = self.requests_per_hour
            self.last_hour_refill = time.time()

        # Check day limit
        if self.day_tokens <= 0:
            sleep_time = 86400 - (time.time() - self.last_day_refill)
            if sleep_time > 0:
                logger.warning(
                    f"Daily rate limit reached (10000/day). Waiting {sleep_time:.1f}s "
                    f"({sleep_time / 3600:.1f} hours)..."
                )
                time.sleep(sleep_time)
            self.day_tokens = self.requests_per_day
            self.last_day_refill = time.time()

        # Check month limit - raise error instead of waiting ~30 days
        if self.month_tokens <= 0:
            raise OpenMeteoRateLimitError(
                "Monthly rate limit exceeded (300,000 calls/month). "
                "Please wait until next month or use a paid API key."
            )


class OpenMeteoWeatherDataFetcher:
    """
    Fetcher for OpenMeteo weather data that returns gridded xarray Datasets.

    It has to:
    - fetch forecast data from OpenMeteo Forecast API for multiple NWP models.
    - fetch historical data from OpenMeteo Archive API (ERA5, etc.).
    - generate lat/lon grids at configurable resolution.
    - return xarray Datasets with latitude, longitude, and time_utc coordinates.
    - cache fetched data as zarr files for efficient reuse.
    - handle API rate limiting and retry failed requests.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        rate_limit_requests_per_minute: int = 600,
        rate_limit_requests_per_hour: int = 5000,
        rate_limit_requests_per_day: int = 10000,
        rate_limit_requests_per_month: int = 300000,
        max_workers: int = 4,
        timeout_seconds: int = 30,
        batch_size: int = 50,
    ):
        """
        Initialize the OpenMeteo data fetcher.

        Args:
            cache_dir: Directory to cache fetched data as zarr files.
                      If None, no caching is performed.
            rate_limit_requests_per_minute: Max API requests per minute (default: 600).
            rate_limit_requests_per_hour: Max API requests per hour (default: 5000).
            rate_limit_requests_per_day: Max API requests per day (default: 10000).
            rate_limit_requests_per_month: Max API requests per month (default: 300000).
            max_workers: Number of parallel workers for batched requests.
            timeout_seconds: HTTP request timeout in seconds.
            batch_size: Number of coordinates per API request (max 50).
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.batch_size = min(batch_size, 50)  # OpenMeteo limit

        self._rate_limiter = RateLimiter(
            requests_per_minute=rate_limit_requests_per_minute,
            requests_per_hour=rate_limit_requests_per_hour,
            requests_per_day=rate_limit_requests_per_day,
            requests_per_month=rate_limit_requests_per_month,
        )

        # Create cache directory if specified
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_forecast(
        self,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        resolution: float = 1.0,
        hourly_parameters: list[str] | None = None,
        forecast_days: int = 7,
        nwp_model: str = "best_match",
        past_days: int = 0,
    ) -> xr.Dataset:
        """
        Fetch forecast data for a lat/lon grid.

        Args:
            lat_range: (min_lat, max_lat) in degrees (-90 to 90).
            lon_range: (min_lon, max_lon) in degrees (-180 to 180).
            resolution: Grid resolution in degrees (default: 1.0).
            hourly_parameters: List of weather variables to fetch.
                             If None, uses DEFAULT_HOURLY_PARAMETERS.
            forecast_days: Number of forecast days (1-16).
            nwp_model: NWP model to use (default: 'best_match').
                      Options: gfs_seamless, ecmwf_ifs04, icon_global, etc.
            past_days: Number of past days to include (0-92).

        Returns:
            xarray.Dataset with dimensions (latitude, longitude, time_utc)
            and requested weather variables as data variables.

        Raises:
            OpenMeteoAPIError: If API request fails.
            OpenMeteoDataError: If data assembly fails.
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        self._validate_lat_lon_range(lat_range, lon_range)
        if forecast_days < 1 or forecast_days > 16:
            raise ValueError("forecast_days must be between 1 and 16")
        if past_days < 0 or past_days > 92:
            raise ValueError("past_days must be between 0 and 92")
        if nwp_model not in FORECAST_NWP_MODELS:
            logger.warning(f"Unknown nwp_model '{nwp_model}', using anyway")

        parameters = hourly_parameters or DEFAULT_HOURLY_PARAMETERS

        # Generate cache key and check cache
        cache_key = self._generate_cache_key(
            endpoint_type="forecast",
            lat_range=lat_range,
            lon_range=lon_range,
            resolution=resolution,
            parameters=parameters,
            start_date=f"past_{past_days}",
            end_date=f"future_{forecast_days}",
            nwp_model=nwp_model,
        )

        cached = self._load_cached(cache_key)
        if cached is not None:
            logger.info("Loaded data from cache")
            return cached

        # Generate grid
        lat_coords, lon_coords, flat_coords = self._generate_grid(lat_range, lon_range, resolution)

        logger.info(
            f"Fetching forecast data for {len(flat_coords)} grid points "
            f"({len(lat_coords)}x{len(lon_coords)} grid)"
        )

        # Build request parameters
        params = {
            "hourly": ",".join(parameters),
            "forecast_days": forecast_days,
            "past_days": past_days,
            "models": nwp_model,
            "timezone": "UTC",
        }

        # Fetch data
        responses = self._fetch_all_coordinates(flat_coords, FORECAST_URL, params)

        # Assemble dataset
        ds = self._assemble_dataset(responses, lat_coords, lon_coords, parameters)

        # Add metadata
        ds.attrs["source"] = "open-meteo"
        ds.attrs["nwp_model"] = nwp_model
        ds.attrs["endpoint"] = "forecast"
        ds.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        ds.attrs["resolution_degrees"] = resolution

        # Cache result
        self._cache_dataset(ds, cache_key)

        return ds

    def fetch_historical(
        self,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        start_date: str,
        end_date: str,
        resolution: float = 1.0,
        hourly_parameters: list[str] | None = None,
        nwp_model: str = "era5",
    ) -> xr.Dataset:
        """
        Fetch historical data from OpenMeteo Historical API.

        Args:
            lat_range: (min_lat, max_lat) in degrees (-90 to 90).
            lon_range: (min_lon, max_lon) in degrees (-180 to 180).
            start_date: Start date in YYYY-MM-DD format (data from 1940).
            end_date: End date in YYYY-MM-DD format.
            resolution: Grid resolution in degrees (default: 1.0).
            hourly_parameters: List of weather variables to fetch.
                             If None, uses DEFAULT_HOURLY_PARAMETERS.
            nwp_model: NWP model to use (default: 'era5').
                      Options: era5, era5_land, cerra, best_match.

        Returns:
            xarray.Dataset with dimensions (latitude, longitude, time_utc)
            and requested weather variables as data variables.

        Raises:
            OpenMeteoAPIError: If API request fails.
            OpenMeteoDataError: If data assembly fails.
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        self._validate_lat_lon_range(lat_range, lon_range)
        self._validate_dates(start_date, end_date)
        if nwp_model not in HISTORICAL_NWP_MODELS:
            logger.warning(f"Unknown nwp_model '{nwp_model}', using anyway")

        parameters = hourly_parameters or DEFAULT_HOURLY_PARAMETERS

        # Generate cache key and check cache
        cache_key = self._generate_cache_key(
            endpoint_type="historical",
            lat_range=lat_range,
            lon_range=lon_range,
            resolution=resolution,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            nwp_model=nwp_model,
        )

        cached = self._load_cached(cache_key)
        if cached is not None:
            logger.info("Loaded data from cache")
            return cached

        # Generate grid
        lat_coords, lon_coords, flat_coords = self._generate_grid(lat_range, lon_range, resolution)

        logger.info(
            f"Fetching historical data for {len(flat_coords)} grid points "
            f"({len(lat_coords)}x{len(lon_coords)} grid) from {start_date} to {end_date}"
        )

        # Build request parameters
        params = {
            "hourly": ",".join(parameters),
            "start_date": start_date,
            "end_date": end_date,
            "models": nwp_model,
            "timezone": "UTC",
        }

        # Fetch data
        responses = self._fetch_all_coordinates(flat_coords, HISTORICAL_URL, params)

        # Assemble dataset
        ds = self._assemble_dataset(responses, lat_coords, lon_coords, parameters)

        # Add metadata
        ds.attrs["source"] = "open-meteo"
        ds.attrs["nwp_model"] = nwp_model
        ds.attrs["endpoint"] = "historical"
        ds.attrs["start_date"] = start_date
        ds.attrs["end_date"] = end_date
        ds.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        ds.attrs["resolution_degrees"] = resolution

        # Cache result
        self._cache_dataset(ds, cache_key)

        return ds

    def _validate_lat_lon_range(
        self,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
    ) -> None:
        """Validate latitude and longitude ranges."""
        if lat_range[0] >= lat_range[1]:
            raise ValueError("lat_range[0] must be less than lat_range[1]")
        if lon_range[0] >= lon_range[1]:
            raise ValueError("lon_range[0] must be less than lon_range[1]")
        if lat_range[0] < -90 or lat_range[1] > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if lon_range[0] < -180 or lon_range[1] > 180:
            raise ValueError("Longitude must be between -180 and 180")

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """Validate date format and range."""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")

        if start >= end:
            raise ValueError("start_date must be before end_date")

    def _generate_grid(
        self,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        resolution: float,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
        """
        Generate lat/lon grid.

        Args:
            lat_range: (min_lat, max_lat) tuple.
            lon_range: (min_lon, max_lon) tuple.
            resolution: Grid spacing in degrees.

        Returns:
            Tuple of (lat_coords, lon_coords, flat_coordinates):
            - lat_coords: 1D array of latitude values
            - lon_coords: 1D array of longitude values
            - flat_coordinates: List of (lat, lon) tuples for all grid points
        """
        lat_coords = np.arange(lat_range[0], lat_range[1] + resolution / 2, resolution)
        lon_coords = np.arange(lon_range[0], lon_range[1] + resolution / 2, resolution)

        # Clip to exact range
        lat_coords = lat_coords[lat_coords <= lat_range[1]]
        lon_coords = lon_coords[lon_coords <= lon_range[1]]

        # Create meshgrid and flatten
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        flat_coordinates = list(zip(lat_grid.flatten(), lon_grid.flatten()))

        return lat_coords, lon_coords, flat_coordinates

    def _batch_coordinates(
        self,
        coordinates: list[tuple[float, float]],
    ) -> list[list[tuple[float, float]]]:
        """Split coordinates into batches for API requests."""
        batches = []
        for i in range(0, len(coordinates), self.batch_size):
            batches.append(coordinates[i : i + self.batch_size])
        return batches

    def _fetch_all_coordinates(
        self,
        coordinates: list[tuple[float, float]],
        endpoint: str,
        params: dict,
    ) -> list[dict]:
        """
        Fetch data for all coordinates with parallel execution and rate limiting.

        Args:
            coordinates: List of (lat, lon) tuples.
            endpoint: API endpoint URL.
            params: Request parameters.

        Returns:
            List of API responses for each coordinate.
        """
        batches = self._batch_coordinates(coordinates)
        all_responses = []

        logger.info(f"Fetching {len(batches)} batches with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for batch_idx, batch in enumerate(batches):
                self._rate_limiter.acquire()
                future = executor.submit(self._fetch_batch, batch, endpoint, params)
                futures[future] = batch_idx

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_responses = future.result()
                    all_responses.extend(batch_responses)
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Completed batch {batch_idx + 1}/{len(batches)}")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    raise

        return all_responses

    def _fetch_batch(
        self,
        coordinates: list[tuple[float, float]],
        endpoint: str,
        params: dict,
    ) -> list[dict]:
        """
        Fetch data for a batch of coordinates.

        Args:
            coordinates: List of (lat, lon) tuples.
            endpoint: API endpoint URL.
            params: Request parameters.

        Returns:
            List of response dictionaries for each coordinate.
        """
        # Build comma-separated coordinate lists
        lats = ",".join(str(round(c[0], 4)) for c in coordinates)
        lons = ",".join(str(round(c[1], 4)) for c in coordinates)

        batch_params = {**params, "latitude": lats, "longitude": lons}

        response = self._fetch_with_retry(endpoint, batch_params)

        # Handle single vs multiple coordinate response format
        if isinstance(response, list):
            return response
        else:
            # Single coordinate returns dict directly
            return [response]

    def _fetch_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> dict | list:
        """
        Fetch with exponential backoff retry.

        Args:
            url: API endpoint URL.
            params: Request parameters.
            max_retries: Maximum number of retry attempts.
            backoff_factor: Multiplier for exponential backoff.

        Returns:
            Parsed JSON response.

        Raises:
            OpenMeteoAPIError: If all retries fail.
        """
        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        for attempt in range(max_retries):
            try:
                request = urllib.request.Request(
                    full_url,
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode())

            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limited
                    wait_time = backoff_factor**attempt
                    logger.warning(f"Rate limited (429), waiting {wait_time:.1f}s before retry")
                    time.sleep(wait_time)
                elif e.code >= 500:  # Server error
                    wait_time = backoff_factor**attempt
                    logger.warning(
                        f"Server error ({e.code}), waiting {wait_time:.1f}s before retry"
                    )
                    time.sleep(wait_time)
                else:
                    error_body = e.read().decode() if e.fp else ""
                    raise OpenMeteoAPIError(f"HTTP {e.code}: {e.reason}. Body: {error_body}")

            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor**attempt
                    logger.warning(
                        f"Connection error, waiting {wait_time:.1f}s before retry: {e.reason}"
                    )
                    time.sleep(wait_time)
                else:
                    raise OpenMeteoAPIError(f"Connection error: {e.reason}")

            except json.JSONDecodeError as e:
                raise OpenMeteoAPIError(f"Invalid JSON response: {e}")

        raise OpenMeteoAPIError(f"Failed after {max_retries} retries")

    def _assemble_dataset(
        self,
        responses: list[dict],
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        parameters: list[str],
    ) -> xr.Dataset:
        """
        Assemble API responses into properly structured xarray Dataset.

        Args:
            responses: List of API response dictionaries.
            lat_coords: 1D array of latitude coordinates.
            lon_coords: 1D array of longitude coordinates.
            parameters: List of weather variable names.

        Returns:
            xarray.Dataset with dimensions (latitude, longitude, time_utc).

        Raises:
            OpenMeteoDataError: If data assembly fails.
        """
        if not responses:
            raise OpenMeteoDataError("No responses to assemble")

        # Extract time coordinates from first response
        first_response = responses[0]
        if "hourly" not in first_response or "time" not in first_response["hourly"]:
            raise OpenMeteoDataError("Invalid response format: missing hourly.time")

        time_strings = first_response["hourly"]["time"]
        time_coords = pd.to_datetime(time_strings)

        n_lat = len(lat_coords)
        n_lon = len(lon_coords)
        n_time = len(time_coords)

        # Initialize data arrays for each variable
        data_vars = {}
        for param in parameters:
            data_vars[param] = np.full((n_lat, n_lon, n_time), np.nan, dtype=np.float32)

        # Build lookup for grid positions
        lat_to_idx = {round(lat, 4): i for i, lat in enumerate(lat_coords)}
        lon_to_idx = {round(lon, 4): i for i, lon in enumerate(lon_coords)}

        # Fill data arrays from responses
        for response in responses:
            lat = round(response.get("latitude", 0), 4)
            lon = round(response.get("longitude", 0), 4)

            lat_idx = lat_to_idx.get(lat)
            lon_idx = lon_to_idx.get(lon)

            if lat_idx is None or lon_idx is None:
                # Try to find nearest coordinate
                lat_idx = np.argmin(np.abs(lat_coords - lat))
                lon_idx = np.argmin(np.abs(lon_coords - lon))

            hourly_data = response.get("hourly", {})
            for param in parameters:
                if param in hourly_data:
                    values = hourly_data[param]
                    if len(values) == n_time:
                        data_vars[param][lat_idx, lon_idx, :] = values

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                param: (["latitude", "longitude", "time_utc"], data)
                for param, data in data_vars.items()
            },
            coords={
                "latitude": lat_coords,
                "longitude": lon_coords,
                "time_utc": time_coords,
            },
        )

        # Add units from first response if available
        hourly_units = first_response.get("hourly_units", {})
        for param in parameters:
            if param in hourly_units and param in ds:
                ds[param].attrs["units"] = hourly_units[param]

        return ds

    def _generate_cache_key(
        self,
        endpoint_type: str,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        resolution: float,
        parameters: list[str],
        start_date: str,
        end_date: str,
        nwp_model: str,
    ) -> str:
        """Generate unique cache key for request parameters."""
        key_data = (
            f"{endpoint_type}_{lat_range}_{lon_range}_{resolution}_"
            f"{'_'.join(sorted(parameters))}_{start_date}_{end_date}_{nwp_model}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _cache_dataset(self, ds: xr.Dataset, cache_key: str) -> None:
        """Cache dataset as zarr."""
        if self.cache_dir is None:
            return

        cache_path = self.cache_dir / f"openmeteo_{cache_key}.zarr"
        try:
            ds.to_zarr(cache_path, mode="w")
            logger.info(f"Cached dataset to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")

    def _load_cached(self, cache_key: str) -> xr.Dataset | None:
        """Load cached dataset if exists."""
        if self.cache_dir is None:
            return None

        cache_path = self.cache_dir / f"openmeteo_{cache_key}.zarr"
        if cache_path.exists():
            try:
                return xr.open_zarr(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_dir is None:
            logger.warning("No cache directory configured")
            return

        import shutil

        for cache_file in self.cache_dir.glob("openmeteo_*.zarr"):
            try:
                shutil.rmtree(cache_file)
                logger.info(f"Removed cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache {cache_file}: {e}")

    def get_available_parameters(self, endpoint: str = "forecast") -> list[str]:
        """
        Get list of available hourly parameters for an endpoint.

        Args:
            endpoint: Either 'forecast' or 'historical'.

        Returns:
            List of available parameter names.
        """
        # These are commonly available parameters
        # Full list depends on the specific model and endpoint
        _ = endpoint  # Parameter reserved for future endpoint-specific lists
        common_params = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "snow_depth",
            "weather_code",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_10m",
            "wind_direction_100m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
        ]
        return common_params
