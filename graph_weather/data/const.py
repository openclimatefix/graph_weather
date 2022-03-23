"""
Constants for use in normalizing data, etc.

Variances for
1. Analysis file fields
2. GFS Forecast Fields
3. GEFS Reforecast Fields

where the variance is the variance in the 3 hour change for a variable averaged across all lat/lon and pressure levels
and time for (~100 random temporal frames, more the better)

Min/Max/Mean/Stddev for all those plus each type of observation in observation files

"""

ANALYSIS_MEANS = []
ANALYSIS_STD = []
ANALYSIS_MAX = []
ANALYSIS_MIN = []
ANALYSIS_VARIANCE = []

FORECAST_MEANS = []
FORECAST_STD = []
FORECAST_MAX = []
FORECAST_MIN = []
FORECAST_VARIANCE = []

REFORECAST_MEANS = []
REFORECAST_STD = []
REFORECAST_MAX = []
REFORECAST_MIN = []
REFORECAST_VARIANCE = []

LANDSEA_MEAN = {"cl": 0.005749, "cvh": 0.08282, "cvl": 0.1114, "slt": 0.6747, "sr": 0.0001, "tvh": 1.607, "tvl": 1.256, "z": 3.717e+03}
LANDSEA_STD = {"cl": 0.0513, "cvh": 0.2523, "cvl": 0.2887, "slt": 1.185, "sr": 0.0, "tvh": 4.833, "tvl": 3.408, "z": 8.375e+03}

SOLAR_STD = [403.1591444098585]
SOLAR_MEAN = [299.97745340056]
SOLAR_MAX = [1414.8996356465245]
SOLAR_MIN = [0.0]
