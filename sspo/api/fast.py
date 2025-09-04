import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sspo.grid_search import get_best_model_filename
from sspo.registry.load_model import load_xgb_reg
from pydantic import BaseModel
import json
import math
from geopy.distance import geodesic


class FromFrontend(BaseModel):
    map_data: str
    line_segments_df: str
    weather_data: dict
    user_inputs: dict


app = FastAPI()
# best_model_name = get_best_model_filename()
# xgb_reg = load_xgb_reg(best_model_name)
app.state.model = pickle.load(open("model.pkl", "rb"))

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_direction(
    start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> int:
    """Calculates the direction (degrees) from a start to an end point (coordinates system)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [start_lat, start_lon, end_lat, end_lon])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dLon
    )
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360

    return int(round(bearing, 0))


@app.get("/")
def index():
    return {"API loaded": True}


@app.get("/predict")
def predict(
    athlete_weight: int,
    distance: int,
    avg_grade: float,
    max_grade: float,
    elevation_gain: int,
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
    avg_power: int,
    temperature: int,
    wind_speed: int,
    wind_direction: int,
):

    X_pred = pd.DataFrame(
        dict(
            athlete_weight=[np.int8(athlete_weight)],
            distance=[np.int32(distance)],
            avg_grade=[np.float16(avg_grade)],
            max_grade=[np.float16(max_grade)],
            elevation_gain=[np.int16(elevation_gain)],
            start_latitude=[np.float16(start_latitude)],
            start_longitude=[np.float16(start_longitude)],
            end_latitude=[np.float16(end_latitude)],
            end_longitude=[np.float16(end_longitude)],
            avg_power=[np.int16(avg_power)],
            temperature=[np.int8(temperature)],
            wind_speed=[np.int8(wind_speed)],
            wind_direction=[np.int16(wind_direction)],
        )
    )

    y_pred = app.state.model.predict(X_pred)
    y_pred = float(y_pred[0])
    return {"Seconds": y_pred}


@app.post("/predict_df")
def predict_df(request: FromFrontend):

    map_data = request.map_data
    map_data_dict = json.loads(map_data)
    map_data_df = pd.DataFrame(map_data_dict)
    line_segments = request.line_segments_df
    line_segments_dict = json.loads(line_segments)
    line_segments_df = pd.DataFrame(line_segments_dict)
    weather_data = request.weather_data
    user_inputs = request.user_inputs
    distance_split = 500  # Use 500-meter segments to match training data

    line_segments_df["id"] = line_segments_df.cumulative_distance.apply(
        lambda x: int(x / distance_split)
    )
    groupby_df = line_segments_df.groupby(by="id").agg(
        {
            "distance_segment": "sum",
            "lat": "first",
            "lon": "first",
            "lat_next": "last",
            "lon_next": "last",
            "gradient": ["mean", "max"],
        }
    )
    groupby_df.columns = groupby_df.columns.map("_".join)
    groupby_df.loc[:, "map_distance"] = groupby_df.apply(
        lambda data: round(
            geodesic(
                (data.lat_first, data.lon_first),
                (data.lat_next_last, data.lon_next_last),
            ).m,
            2,
        ),
        axis=1,
    )
    groupby_df.loc[:, "map_direction"] = groupby_df.apply(
        lambda data: int(
            round(
                get_direction(
                    data.lat_first,
                    data.lon_first,
                    data.lat_next_last,
                    data.lon_next_last,
                ),
                0,
            )
        ),
        axis=1,
    )
    groupby_df.loc[:, "athlete_weight"] = user_inputs["weight"]
    groupby_df.loc[:, "power_mean"] = user_inputs["power"]
    groupby_df.loc[:, "power_max"] = user_inputs["power"] * 1.85
    groupby_df.loc[:, "speed_first"] = user_inputs["entry_speed"]
    groupby_df.loc[:, "temperature"] = round(weather_data["temperature"], 0)
    groupby_df.loc[:, "wind_speed"] = round(weather_data["wind_speed"], 0)
    groupby_df.loc[:, "wind_direction"] = round(weather_data["wind_direction"], 0)

    times = []
    calc_speed = []
    groupby_df.drop(
        ["lat_first", "lon_first", "lat_next_last", "lon_next_last"],
        axis=1,
        inplace=True,
    )
    groupby_df.rename(columns={"distance_segment_sum": "distance"}, inplace=True)

    expected_order = [
        "athlete_weight",
        "distance",
        "gradient_mean",
        "gradient_max",
        "map_distance",
        "map_direction",
        "power_mean",
        "power_max",
        "speed_first",
        "temperature",
        "wind_speed",
        "wind_direction",
    ]

    groupby_df = groupby_df[expected_order]

    for i, (idx, each) in enumerate(groupby_df.iterrows()):
        each = groupby_df.loc[groupby_df.index == idx].reset_index(drop=True)

        if i == 0:  # First iteration
            y_pred = app.state.model.predict(each)
            y_pred = float(y_pred[0])
            speed = each.distance / y_pred * 3.6
            times.append(y_pred)
            calc_speed.append(speed[0])

        else:
            each["speed_first"] = calc_speed[-1]

            y_pred = app.state.model.predict(each)
            y_pred = float(y_pred[0])
            speed = each.distance / y_pred * 3.6
            calc_speed.append(speed[0])
            times.append(y_pred)

    final_sum = sum(times)

    groupby_df["times"] = times

    dataframe_dict = groupby_df.astype(object).to_dict(orient="records")

    return {"seconds": float(final_sum), "dataframe": dataframe_dict}
