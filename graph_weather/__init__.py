"""Main import for the complete models"""

try:
    from .data.nnja_ai import SensorDataset
except ImportError:
    SensorDataset = None
try:
    from .data.weather_station_reader import WeatherStationReader
except ImportError:
    WeatherStationReader = None
from .models.analysis import GraphWeatherAssimilator
from .models.forecast import GraphWeatherForecaster
