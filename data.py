import os
from stravalib import Client as StravaClient
import numpy as np
import pandas as pd
import openmeteo_requests
#import requests_cache
#from retry_requests import retry

#pip install stravalib
#pip install openmeteo-requests
#pip install requests-cache retry-requests numpy pandas

ACCESS_TOKEN = os.environ["ACCESS_TOKEN"]
SEGMENT_ID_1 = os.environ["SEGMENT_ID_1"]
SEGMENT_1_EFFORT_ID_1 = os.environ["SEGMENT_1_EFFORT_ID_1"]
#ATHLETE_ID_1 = os.environ["ATHLETE_ID_1"]

client = StravaClient(access_token=ACCESS_TOKEN)
segment = client.get_segment(SEGMENT_ID_1)
effort = client.get_segment_effort(SEGMENT_1_EFFORT_ID_1)
athlete = client.get_athlete()

openmeteo = openmeteo_requests.Client()
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": segment.start_latlng.root[0], #52.52,
	"longitude": segment.start_latlng.root[1], #13.41,
	"start_date": f"{effort.start_date_local.year}-{effort.start_date_local.month:02d}-{effort.start_date_local.day:02d}", #"2025-08-09",
	"end_date": f"{effort.start_date_local.year}-{effort.start_date_local.month:02d}-{effort.start_date_local.day:02d}", #"2025-08-23",
	"hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m"],
	"timezone": "auto",
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0]
hourly = response.Hourly()

effort_dict = {'athlete_weight': [athlete.weight],
 'distance': [segment.distance],
 'avg_grade': [segment.average_grade],
 'max_grade': [segment.maximum_grade],
 'elevation_gain': [segment.total_elevation_gain],
 'avg_power': [effort.average_watts],
 'temperature': [hourly.Variables(0).Values(effort.start_date_local.hour)],
 'wind_speed': [hourly.Variables(1).Values(effort.start_date_local.hour)],
 'wind_direction': [hourly.Variables(2).Values(effort.start_date_local.hour)],
 'time': [effort.moving_time]}

effort_dtypes = {'athlete_weight': np.int8,
 'distance': np.int32,
 'avg_grade': np.float16,
 'max_grade': np.float16,
 'elevation_gain': np.int16,
 'avg_power': np.int16,
 'temperature': np.int8,
 'wind_speed': np.int8,
 'wind_direction': np.int16,
 'time': np.int16}

effort_df = pd.DataFrame(effort_dict)
effort_df = effort_df.astype(effort_dtypes)
print(effort_df)

effort_df.to_parquet('parquet_test.parquet')
#parquet
