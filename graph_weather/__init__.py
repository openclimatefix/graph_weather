"""Main import for the complete models"""

# Using lazy loading to avoid dependency conflicts
def __getattr__(name):
    """Lazy loading for all modules to avoid dependency conflicts."""
    if name == "GraphWeatherAssimilator":
        from .models.analysis import GraphWeatherAssimilator as GWA
        globals()[name] = GWA
        return GWA
    elif name == "GraphWeatherForecaster":
        from .models.forecast import GraphWeatherForecaster as GWF
        globals()[name] = GWF
        return GWF
    elif name == "SensorDataset":
        from .data.nnja_ai import SensorDataset as SD
        globals()[name] = SD
        return SD
    elif name == "WeatherStationReader":
        from .data.weather_station_reader import WeatherStationReader as WSR
        globals()[name] = WSR
        return WSR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GraphWeatherAssimilator",
    "GraphWeatherForecaster",
    "SensorDataset",
    "WeatherStationReader",
]
