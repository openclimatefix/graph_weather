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

