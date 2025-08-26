import os
from stravalib import Client as StravaClient
import numpy as np
import pandas as pd
import openmeteo_requests
from pathlib import Path
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
	"latitude": segment.start_latlng.root[0],
	"longitude": segment.start_latlng.root[1],
	"start_date": f"{effort.start_date_local.year}-{effort.start_date_local.month:02d}-{effort.start_date_local.day:02d}",
	"end_date": f"{effort.start_date_local.year}-{effort.start_date_local.month:02d}-{effort.start_date_local.day:02d}",
	"hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m"],
	"timezone": "auto",
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0]
hourly = response.Hourly()

effort_dict = {'id': [effort.id],
 'athlete_weight': [athlete.weight],
 'distance': [segment.distance],
 'avg_grade': [round(segment.average_grade, 1)],
 'max_grade': [round(segment.maximum_grade, 1)],
 'elevation_gain': [segment.total_elevation_gain],
 "start_latitude": [segment.start_latlng.root[0]],
 "start_longitude": [segment.start_latlng.root[1]],
 "end_latitude": [segment.end_latlng.root[0]],
 "end_longitude": [segment.end_latlng.root[1]],
 'avg_power': [effort.average_watts],
 'temperature': [hourly.Variables(0).Values(effort.start_date_local.hour)],
 'wind_speed': [hourly.Variables(1).Values(effort.start_date_local.hour)],
 'wind_direction': [hourly.Variables(2).Values(effort.start_date_local.hour)],
 'time': [effort.moving_time]}

effort_dtypes = {'id': np.int64,
 'athlete_weight': np.int8,
 'distance': np.int32,
 'avg_grade': np.float16,
 'max_grade': np.float16,
 'elevation_gain': np.int16,
 "start_latitude": np.float16,
 "start_longitude": np.float16,
 "end_latitude": np.float16,
 "end_longitude": np.float16,
 'avg_power': np.int16,
 'temperature': np.int8,
 'wind_speed': np.int8,
 'wind_direction': np.int16,
 'time': np.int16}

effort_df = pd.DataFrame(effort_dict)
effort_df = effort_df.astype(effort_dtypes)
print(effort_df)

file_path = Path('parquet_test.parquet')
effort_df.to_parquet(file_path, engine='fastparquet')
