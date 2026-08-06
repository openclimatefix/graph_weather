"""Tests for OpenMeteoWeatherDataFetcher."""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch

from graph_weather.data.openmeteo_fetcher import (
    OpenMeteoWeatherDataFetcher,
    OpenMeteoError,
    OpenMeteoAPIError,
    OpenMeteoDataError,
    OpenMeteoRateLimitError,
    RateLimiter,
    DEFAULT_HOURLY_PARAMETERS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fetcher(tmp_path):
    """Create fetcher with temp cache dir."""
    return OpenMeteoWeatherDataFetcher(
        cache_dir=str(tmp_path),
        max_workers=2,
        timeout_seconds=10,
    )


@pytest.fixture
def fetcher_no_cache():
    """Create fetcher without cache."""
    return OpenMeteoWeatherDataFetcher(
        cache_dir=None,
        max_workers=2,
    )


@pytest.fixture
def mock_api_response():
    """Create mock API response for a single coordinate."""
    return {
        "latitude": 52.52,
        "longitude": 13.41,
        "elevation": 38.0,
        "timezone": "UTC",
        "hourly": {
            "time": [
                "2024-01-01T00:00",
                "2024-01-01T01:00",
                "2024-01-01T02:00",
            ],
            "temperature_2m": [5.2, 4.8, 4.5],
            "relative_humidity_2m": [85, 87, 88],
            "pressure_msl": [1013.2, 1013.5, 1013.8],
        },
        "hourly_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "pressure_msl": "hPa",
        },
    }


# ============================================================================
# RateLimiter Tests
# ============================================================================


def test_rate_limiter_initialization():
    """Test rate limiter initializes correctly with all limits."""
    limiter = RateLimiter(
        requests_per_minute=60,
        requests_per_hour=500,
        requests_per_day=1000,
        requests_per_month=10000,
    )
    assert limiter.requests_per_minute == 60
    assert limiter.requests_per_hour == 500
    assert limiter.requests_per_day == 1000
    assert limiter.requests_per_month == 10000


def test_rate_limiter_default_limits():
    """Test default limits match OpenMeteo free tier."""
    limiter = RateLimiter()
    assert limiter.requests_per_minute == 600
    assert limiter.requests_per_hour == 5000
    assert limiter.requests_per_day == 10000
    assert limiter.requests_per_month == 300000


def test_rate_limiter_acquire_consumes_tokens():
    """Test that acquire consumes a token from all buckets."""
    limiter = RateLimiter(
        requests_per_minute=10,
        requests_per_hour=50,
        requests_per_day=100,
        requests_per_month=500,
    )
    initial_minute = limiter.minute_tokens
    initial_hour = limiter.hour_tokens
    initial_day = limiter.day_tokens
    initial_month = limiter.month_tokens

    limiter.acquire()

    assert limiter.minute_tokens == initial_minute - 1
    assert limiter.hour_tokens == initial_hour - 1
    assert limiter.day_tokens == initial_day - 1
    assert limiter.month_tokens == initial_month - 1


def test_rate_limiter_month_limit_raises_error():
    """Test that monthly limit raises error instead of waiting."""
    limiter = RateLimiter(
        requests_per_minute=100,
        requests_per_hour=100,
        requests_per_day=100,
        requests_per_month=1,
    )
    limiter.acquire()
    with pytest.raises(OpenMeteoRateLimitError, match="Monthly rate limit"):
        limiter.acquire()


# ============================================================================
# OpenMeteoWeatherDataFetcher Initialization Tests
# ============================================================================


def test_fetcher_initialization(fetcher):
    """Test fetcher initializes correctly."""
    assert fetcher.max_workers == 2
    assert fetcher.timeout_seconds == 10
    assert fetcher.batch_size == 50


def test_fetcher_batch_size_limit():
    """Test batch size is capped at 50."""
    fetcher = OpenMeteoWeatherDataFetcher(batch_size=100)
    assert fetcher.batch_size == 50


def test_fetcher_no_cache(fetcher_no_cache):
    """Test fetcher initializes without cache."""
    assert fetcher_no_cache.cache_dir is None


# ============================================================================
# Grid Generation Tests
# ============================================================================


def test_grid_generation(fetcher):
    """Test lat/lon grid generation."""
    lat_coords, lon_coords, flat_coords = fetcher._generate_grid(
        lat_range=(50.0, 52.0),
        lon_range=(10.0, 12.0),
        resolution=1.0,
    )

    assert len(lat_coords) == 3
    assert len(lon_coords) == 3
    assert len(flat_coords) == 9

    np.testing.assert_array_almost_equal(lat_coords, [50.0, 51.0, 52.0])
    np.testing.assert_array_almost_equal(lon_coords, [10.0, 11.0, 12.0])


def test_grid_generation_fractional_resolution(fetcher):
    """Test grid generation with fractional resolution."""
    lat_coords, lon_coords, flat_coords = fetcher._generate_grid(
        lat_range=(50.0, 51.0),
        lon_range=(10.0, 11.0),
        resolution=0.5,
    )

    assert len(lat_coords) == 3
    assert len(lon_coords) == 3
    assert len(flat_coords) == 9


# ============================================================================
# Coordinate Batching Tests
# ============================================================================


def test_batch_coordinates(fetcher):
    """Test coordinate batching."""
    coords = [(i, j) for i in range(10) for j in range(12)]
    batches = fetcher._batch_coordinates(coords)

    assert len(batches) == 3
    assert len(batches[0]) == 50
    assert len(batches[1]) == 50
    assert len(batches[2]) == 20


def test_batch_coordinates_small(fetcher):
    """Test batching with fewer coordinates than batch size."""
    coords = [(i, j) for i in range(3) for j in range(3)]
    batches = fetcher._batch_coordinates(coords)

    assert len(batches) == 1
    assert len(batches[0]) == 9


# ============================================================================
# Validation Tests
# ============================================================================


def test_validate_lat_lon_range_valid(fetcher):
    """Test validation passes for valid ranges."""
    fetcher._validate_lat_lon_range(
        lat_range=(-45.0, 45.0),
        lon_range=(-90.0, 90.0),
    )


def test_validate_lat_lon_range_invalid_lat_order(fetcher):
    """Test validation fails when lat min >= max."""
    with pytest.raises(ValueError, match="lat_range.*less than"):
        fetcher._validate_lat_lon_range(
            lat_range=(45.0, -45.0),
            lon_range=(-90.0, 90.0),
        )


def test_validate_lat_lon_range_invalid_lon_order(fetcher):
    """Test validation fails when lon min >= max."""
    with pytest.raises(ValueError, match="lon_range.*less than"):
        fetcher._validate_lat_lon_range(
            lat_range=(-45.0, 45.0),
            lon_range=(90.0, -90.0),
        )


def test_validate_lat_lon_range_out_of_bounds(fetcher):
    """Test validation fails for out of bounds coordinates."""
    with pytest.raises(ValueError, match="Latitude.*-90.*90"):
        fetcher._validate_lat_lon_range(
            lat_range=(-100.0, 45.0),
            lon_range=(-90.0, 90.0),
        )


def test_validate_dates_valid(fetcher):
    """Test date validation passes for valid dates."""
    fetcher._validate_dates("2023-01-01", "2023-12-31")


def test_validate_dates_invalid_format(fetcher):
    """Test date validation fails for invalid format."""
    with pytest.raises(ValueError, match="Invalid date format"):
        fetcher._validate_dates("not-a-date", "2023-12-31")


def test_validate_dates_start_after_end(fetcher):
    """Test date validation fails when start >= end."""
    with pytest.raises(ValueError, match="start_date must be before"):
        fetcher._validate_dates("2023-12-31", "2023-01-01")


# ============================================================================
# Cache Key Tests
# ============================================================================


def test_cache_key_generation(fetcher):
    """Test cache key generation is deterministic."""
    key1 = fetcher._generate_cache_key(
        endpoint_type="forecast",
        lat_range=(50.0, 52.0),
        lon_range=(10.0, 12.0),
        resolution=1.0,
        parameters=["temperature_2m", "pressure_msl"],
        start_date="2024-01-01",
        end_date="2024-01-07",
        nwp_model="best_match",
    )

    key2 = fetcher._generate_cache_key(
        endpoint_type="forecast",
        lat_range=(50.0, 52.0),
        lon_range=(10.0, 12.0),
        resolution=1.0,
        parameters=["temperature_2m", "pressure_msl"],
        start_date="2024-01-01",
        end_date="2024-01-07",
        nwp_model="best_match",
    )

    assert key1 == key2
    assert len(key1) == 16


def test_cache_key_different_params(fetcher):
    """Test cache keys differ for different parameters."""
    key1 = fetcher._generate_cache_key(
        endpoint_type="forecast",
        lat_range=(50.0, 52.0),
        lon_range=(10.0, 12.0),
        resolution=1.0,
        parameters=["temperature_2m"],
        start_date="2024-01-01",
        end_date="2024-01-07",
        nwp_model="best_match",
    )

    key2 = fetcher._generate_cache_key(
        endpoint_type="forecast",
        lat_range=(50.0, 52.0),
        lon_range=(10.0, 12.0),
        resolution=1.0,
        parameters=["pressure_msl"],
        start_date="2024-01-01",
        end_date="2024-01-07",
        nwp_model="best_match",
    )

    assert key1 != key2


# ============================================================================
# Dataset Assembly Tests
# ============================================================================


def test_assemble_dataset_structure(fetcher, mock_api_response):
    """Test _assemble_dataset creates correct xarray structure."""
    responses = []
    for lat in [52.0, 53.0]:
        for lon in [13.0, 14.0]:
            response = mock_api_response.copy()
            response["latitude"] = lat
            response["longitude"] = lon
            response["hourly"] = mock_api_response["hourly"].copy()
            responses.append(response)

    lat_coords = np.array([52.0, 53.0])
    lon_coords = np.array([13.0, 14.0])
    parameters = ["temperature_2m", "relative_humidity_2m"]

    ds = fetcher._assemble_dataset(responses, lat_coords, lon_coords, parameters)

    assert isinstance(ds, xr.Dataset)
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time_utc" in ds.coords
    assert len(ds.latitude) == 2
    assert len(ds.longitude) == 2
    assert len(ds.time_utc) == 3
    assert "temperature_2m" in ds.data_vars
    assert "relative_humidity_2m" in ds.data_vars
    assert ds.temperature_2m.dims == ("latitude", "longitude", "time_utc")


def test_assemble_dataset_empty_responses(fetcher):
    """Test _assemble_dataset raises error for empty responses."""
    with pytest.raises(OpenMeteoDataError, match="No responses"):
        fetcher._assemble_dataset(
            responses=[],
            lat_coords=np.array([52.0]),
            lon_coords=np.array([13.0]),
            parameters=["temperature_2m"],
        )


def test_assemble_dataset_invalid_response(fetcher):
    """Test _assemble_dataset raises error for invalid response format."""
    invalid_response = {"latitude": 52.0, "longitude": 13.0}

    with pytest.raises(OpenMeteoDataError, match="Invalid response format"):
        fetcher._assemble_dataset(
            responses=[invalid_response],
            lat_coords=np.array([52.0]),
            lon_coords=np.array([13.0]),
            parameters=["temperature_2m"],
        )


# ============================================================================
# Fetch Forecast Tests
# ============================================================================


@patch.object(OpenMeteoWeatherDataFetcher, "_fetch_all_coordinates")
def test_fetch_forecast_returns_xarray(mock_fetch_all, fetcher, mock_api_response):
    """Test fetch_forecast returns properly structured xarray Dataset."""
    mock_fetch_all.return_value = [mock_api_response]

    ds = fetcher.fetch_forecast(
        lat_range=(52.0, 52.5),
        lon_range=(13.0, 13.5),
        resolution=0.5,
        hourly_parameters=["temperature_2m", "relative_humidity_2m"],
        forecast_days=1,
    )

    assert isinstance(ds, xr.Dataset)
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time_utc" in ds.coords
    assert ds.attrs["source"] == "open-meteo"
    assert ds.attrs["endpoint"] == "forecast"


@patch.object(OpenMeteoWeatherDataFetcher, "_fetch_all_coordinates")
def test_fetch_forecast_gridded_output(mock_fetch_all, fetcher):
    """Test that output is gridded data, not single point (PR #93 fix)."""
    responses = []
    for lat in [50.0, 51.0, 52.0]:
        for lon in [10.0, 11.0, 12.0]:
            responses.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": {
                        "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                        "temperature_2m": [10.0, 11.0],
                    },
                    "hourly_units": {"temperature_2m": "°C"},
                }
            )

    mock_fetch_all.return_value = responses

    ds = fetcher.fetch_forecast(
        lat_range=(50.0, 52.0),
        lon_range=(10.0, 12.0),
        resolution=1.0,
        hourly_parameters=["temperature_2m"],
        forecast_days=1,
    )

    assert len(ds.latitude) > 1, "Must return gridded data, not single point"
    assert len(ds.longitude) > 1, "Must return gridded data, not single point"

    for var in ds.data_vars:
        assert len(ds[var].dims) == 3, f"Variable {var} should be 3D"


@patch.object(OpenMeteoWeatherDataFetcher, "_fetch_all_coordinates")
def test_fetch_forecast_configurable_parameters(mock_fetch_all, fetcher):
    """Test that hourly parameters are configurable (maintainer requirement)."""
    custom_params = ["temperature_2m", "wind_speed_10m"]

    mock_fetch_all.return_value = [
        {
            "latitude": 52.0,
            "longitude": 13.0,
            "hourly": {
                "time": ["2024-01-01T00:00"],
                "temperature_2m": [10.0],
                "wind_speed_10m": [5.0],
            },
            "hourly_units": {},
        }
    ]

    ds = fetcher.fetch_forecast(
        lat_range=(52.0, 52.5),
        lon_range=(13.0, 13.5),
        resolution=1.0,
        hourly_parameters=custom_params,
        forecast_days=1,
    )

    assert "temperature_2m" in ds.data_vars
    assert "wind_speed_10m" in ds.data_vars
    assert "relative_humidity_2m" not in ds.data_vars


def test_fetch_forecast_invalid_forecast_days(fetcher):
    """Test fetch_forecast raises error for invalid forecast_days."""
    with pytest.raises(ValueError, match="forecast_days.*1.*16"):
        fetcher.fetch_forecast(
            lat_range=(50.0, 52.0),
            lon_range=(10.0, 12.0),
            forecast_days=20,
        )


def test_fetch_forecast_invalid_past_days(fetcher):
    """Test fetch_forecast raises error for invalid past_days."""
    with pytest.raises(ValueError, match="past_days.*0.*92"):
        fetcher.fetch_forecast(
            lat_range=(50.0, 52.0),
            lon_range=(10.0, 12.0),
            past_days=-1,
        )


# ============================================================================
# Fetch Historical Tests
# ============================================================================


@patch.object(OpenMeteoWeatherDataFetcher, "_fetch_all_coordinates")
def test_fetch_historical_returns_xarray(mock_fetch_all, fetcher, mock_api_response):
    """Test fetch_historical returns properly structured xarray Dataset."""
    mock_fetch_all.return_value = [mock_api_response]

    ds = fetcher.fetch_historical(
        lat_range=(52.0, 52.5),
        lon_range=(13.0, 13.5),
        start_date="2023-01-01",
        end_date="2023-01-02",
        resolution=0.5,
        hourly_parameters=["temperature_2m"],
    )

    assert isinstance(ds, xr.Dataset)
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time_utc" in ds.coords
    assert ds.attrs["source"] == "open-meteo"
    assert ds.attrs["endpoint"] == "historical"
    assert ds.attrs["start_date"] == "2023-01-01"
    assert ds.attrs["end_date"] == "2023-01-02"


# ============================================================================
# Caching Tests
# ============================================================================


@patch.object(OpenMeteoWeatherDataFetcher, "_fetch_all_coordinates")
def test_caching_saves_and_loads(mock_fetch_all, fetcher, mock_api_response, tmp_path):
    """Test data caching functionality."""
    mock_fetch_all.return_value = [mock_api_response]

    fetcher.fetch_forecast(
        lat_range=(52.0, 52.5),
        lon_range=(13.0, 13.5),
        resolution=0.5,
        hourly_parameters=["temperature_2m"],
        forecast_days=1,
    )

    cache_files = list(tmp_path.glob("openmeteo_*.zarr"))
    assert len(cache_files) == 1

    mock_fetch_all.reset_mock()
    fetcher.fetch_forecast(
        lat_range=(52.0, 52.5),
        lon_range=(13.0, 13.5),
        resolution=0.5,
        hourly_parameters=["temperature_2m"],
        forecast_days=1,
    )

    mock_fetch_all.assert_not_called()


def test_caching_disabled(fetcher_no_cache):
    """Test that caching can be disabled."""
    cache_key = "test_key"
    fetcher_no_cache._cache_dataset(xr.Dataset(), cache_key)
    result = fetcher_no_cache._load_cached(cache_key)
    assert result is None


def test_clear_cache(fetcher, tmp_path):
    """Test cache clearing."""
    (tmp_path / "openmeteo_abc123.zarr").mkdir()
    (tmp_path / "openmeteo_def456.zarr").mkdir()

    fetcher.clear_cache()

    cache_files = list(tmp_path.glob("openmeteo_*.zarr"))
    assert len(cache_files) == 0


# ============================================================================
# Utility Method Tests
# ============================================================================


def test_get_available_parameters(fetcher):
    """Test getting available parameters."""
    params = fetcher.get_available_parameters()

    assert isinstance(params, list)
    assert len(params) > 0
    assert "temperature_2m" in params
    assert "pressure_msl" in params


def test_default_hourly_parameters():
    """Test default hourly parameters are defined."""
    assert len(DEFAULT_HOURLY_PARAMETERS) > 0
    assert "temperature_2m" in DEFAULT_HOURLY_PARAMETERS
    assert "pressure_msl" in DEFAULT_HOURLY_PARAMETERS


# ============================================================================
# Exception Tests
# ============================================================================


def test_openmeteo_error_base():
    """Test base exception can be raised."""
    with pytest.raises(OpenMeteoError):
        raise OpenMeteoError("Test error")


def test_openmeteo_api_error():
    """Test API error can be raised."""
    with pytest.raises(OpenMeteoAPIError):
        raise OpenMeteoAPIError("API failed")


def test_openmeteo_data_error():
    """Test data error can be raised."""
    with pytest.raises(OpenMeteoDataError):
        raise OpenMeteoDataError("Data assembly failed")


def test_exception_inheritance():
    """Test exception inheritance works correctly."""
    with pytest.raises(OpenMeteoError):
        raise OpenMeteoAPIError("Should be caught as base")

    with pytest.raises(OpenMeteoError):
        raise OpenMeteoDataError("Should be caught as base")
