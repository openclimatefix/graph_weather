import openmeteo_requests  # Importing required libraries
import requests_cache
import pandas as pd
from retry_requests import retry

class WeatherDataFetcher:
    BASE_URL = "https://api.open-meteo.com/v1/"  # Base URL for OpenMeteo API

    def __init__(self):
        # Initialize the WeatherDataFetcher class
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_forecast_data(self, NWP, params):
        # Fetch weather data from OpenMeteo API for the specified model (NWP) and parameters
        url = f"https://api.open-meteo.com/v1/{NWP}"  # Construct API URL
        try:
            responses = self.openmeteo.weather_api(url, params=params)  # Get weather data
            return responses[0]  # Return the first response (assuming only one location)
        except openmeteo_requests.OpenMeteoRequestsError as e:
            # Handle OpenMeteoRequestsError exceptions
            if 'No data is available for this location' in str(e):
                print(f"Error: No data available for the location for model '{NWP}'.")
            else:
                print(f"Error: {e}")
            return None
        
    def fetch_historical_data(self, params):
        # Fetch historical weather data from OpenMeteo API
        BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
        try:
            responses = self.openmeteo.weather_api(BASE_URL, params=params)
            return responses[0] if responses else None
        except ValueError as e:
            print(f"Error: {e}")
            return None

    def process_hourly_data(self, response):
        # Process hourly data from OpenMeteo API response
        # Extract hourly data from the response
        hourly = response.Hourly()

        # Extract variables
        hourly_variables = {
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "precipitation": hourly.Variables(2).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(3).ValuesAsNumpy()
        }

        # Extract time information
        time_range = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        # Create a dictionary for hourly data
        hourly_data = {"date": time_range}

        # Assign each variable to the corresponding key in the dictionary
        for variable_name, variable_values in hourly_variables.items():
            hourly_data[variable_name] = variable_values

        # Create a DataFrame from the dictionary
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        return hourly_dataframe

    def print_location_info(self, response):
        # Print location information from OpenMeteo API response
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")


def main():
    # Main function to demonstrate usage of WeatherDataFetcher class
    fetcher = WeatherDataFetcher()  # Create instance of WeatherDataFetcher

    # Specify parameters for weather data fetch
    NWP = "gfs"  # Choose NWP model
    
    # NWP models =  ["dwd-icon", "gfs", "ecmwf", "meteofrance", "jma", "metno", "gem", "bom", "cma"]

    params = {
        "latitude": 40.77,  # Latitude of the location
        "longitude": -73.91,  # Longitude of the location
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover"],  # Variables to fetch
        "start_date": "2023-12-21",  # Start date for data
        "end_date": "2024-03-15"  # End date for data
    }

    # Fetch weather data for specified model and parameters
    response = fetcher.fetch_forecast_data(NWP, params)

    # Print location information
    fetcher.print_location_info(response)

    # Process and print hourly data
    gfs_dataframe = fetcher.process_hourly_data(response)
    print(gfs_dataframe)

    # Fetch historical weather data
    history = fetcher.fetch_historical_data(params)
    history_dataframe = fetcher.process_hourly_data(history)
    print(history_dataframe)


if __name__ == "__main__":
    main()  # Call main function if script is executed directly
