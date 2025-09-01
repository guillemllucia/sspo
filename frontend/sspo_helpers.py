# sspo_helpers.py
import streamlit as st
import pandas as pd
import numpy as np
import polyline
import requests
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, time, timedelta
import math
import time as time_sleep
from bs4 import BeautifulSoup

# --- DATA FETCHING & PREDICTION ---
@st.cache_data
def get_segment_data(_segment_id, access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    url = f"https://www.strava.com/api/v3/segments/{_segment_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        encoded_polyline = data.get('map', {}).get('polyline')
        if encoded_polyline:
            decoded_coords = polyline.decode(encoded_polyline)
            map_df = pd.DataFrame(decoded_coords, columns=['lat', 'lon'])
            return {
                "name": data.get("name", "N/A"),
                "map_data": map_df,
                "start_latlng": data.get("start_latlng", [0, 0]),
                "distance": data.get("distance", 0),
                "elevation_gain": data.get("total_elevation_gain", 0)
            }
        return None
    except requests.exceptions.RequestException:
        return None

@st.cache_data
def get_weather_forecast(lat, lon, ride_datetime):
    url = "https://api.open-meteo.com/v1/forecast"
    ride_date_str = ride_datetime.strftime('%Y-%m-%d')
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m", "wind_speed_unit": "ms", "timezone": "auto", "start_date": ride_date_str, "end_date": ride_date_str}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        target_hour_str = ride_datetime.strftime('%Y-%m-%dT%H:00')
        hour_index = data['hourly']['time'].index(target_hour_str)
        return {"temperature": data['hourly']['temperature_2m'][hour_index], "wind_speed": data['hourly']['wind_speed_10m'][hour_index] * 3.6, "wind_direction": data['hourly']['wind_direction_10m'][hour_index]}
    except (requests.exceptions.RequestException, ValueError, IndexError):
        return None

@st.cache_data
def get_elevation_data(map_df):
    url = "https://api.open-meteo.com/v1/elevation"
    all_elevations = []
    for i in range(0, len(map_df), 100):
        batch = map_df.iloc[i:i+100]
        params = {"latitude": ",".join(batch['lat'].astype(str)), "longitude": ",".join(batch['lon'].astype(str))}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            all_elevations.extend(response.json()['elevation'])
            time_sleep.sleep(0.5) # Shorter delay
        except requests.exceptions.RequestException:
            return None
    map_df['elevation'] = all_elevations
    return map_df

def predict_time_from_api(segment_data, map_data, line_segments_df, weather_data, user_inputs):
    api_url = "https://api-879488749692.europe-west1.run.app/predict"
    try:
        params = {
            'athlete_weight': int(user_inputs['weight']),
            'distance': int(segment_data['distance']),
            'avg_grade': round(float((map_data['elevation'].iloc[-1] - map_data['elevation'].iloc[0]) / segment_data['distance'] * 100 if segment_data['distance'] > 0 else 0), 4),
            'max_grade': round(float(line_segments_df['gradient'].max()), 4),
            'elevation_gain': int(segment_data['elevation_gain']),
            'start_latitude': round(float(segment_data['start_latlng'][0]), 6),
            'start_longitude': round(float(segment_data['start_latlng'][1]), 6),
            'end_latitude': round(float(map_data['lat'].iloc[-1]), 6),
            'end_longitude': round(float(map_data['lon'].iloc[-1]), 6),
            'avg_power': int(user_inputs['power']),
            'temperature': int(weather_data['temperature']),
            'wind_speed': int(weather_data['wind_speed']),
            'wind_direction': int(weather_data['wind_direction']),
        }
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        prediction_data = response.json()
        return int(prediction_data['Seconds']), params
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while calling the prediction API: {e}")
        st.error(f"API Response: {e.response.text}")
        return None, {}
    except (KeyError, IndexError):
        st.error("Received an unexpected response from the prediction API.")
        return None, {}

def find_power_for_target_time(target_seconds, segment_data, map_data, line_segments_df, weather_data, user_inputs):
    """Iteratively finds the power required to meet a target time."""
    current_power = user_inputs['power']

    with st.spinner("Calculating power needed for Top 10..."):
        for _ in range(20): # Limit iterations to prevent infinite loops
            inputs_for_calc = user_inputs.copy()
            inputs_for_calc['power'] = current_power

            predicted_time, _ = predict_time_from_api(segment_data, map_data, line_segments_df, weather_data, inputs_for_calc)

            if predicted_time is None:
                return None # API error occurred

            if predicted_time <= target_seconds:
                return current_power # Success!

            time_diff = predicted_time - target_seconds
            power_adjustment = max(1, int(time_diff / 5))
            current_power += power_adjustment

            if current_power > 1000: # Safety break
                return None

    return None # Could not find a suitable power


# --- CALCULATION HELPERS ---
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a)) * 1000

def get_color_from_gradient(gradient):
    if gradient > 8: return [204, 0, 0, 200]
    elif gradient > 5: return [255, 0, 0, 200]
    elif gradient > 2: return [255, 128, 0, 200]
    elif gradient > -2: return [0, 255, 0, 200]
    else: return [0, 0, 255, 200]

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def get_wind_description(segment_bearing, wind_direction):
    diff = abs(segment_bearing - wind_direction)
    angle = min(diff, 360 - diff)
    if angle <= 45: return "Tailwind"
    elif angle >= 135: return "Headwind"
    else: return "Crosswind"

def degrees_to_cardinal(d):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

def parse_time_to_seconds(time_str):
    if pd.isna(time_str): return None
    time_str = str(time_str)
    total_seconds = 0
    if 'h' in time_str:
        parts = time_str.split('h'); total_seconds += int(parts[0]) * 3600; time_str = parts[1].strip()
    if 'm' in time_str:
        parts = time_str.split('m'); total_seconds += int(parts[0]) * 60; time_str = parts[1].strip()
    if 's' in time_str:
        total_seconds += int(time_str.replace('s', '').strip())
    elif ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2: total_seconds += int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: total_seconds += int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return total_seconds if total_seconds > 0 else None

# --- VISUALIZATION COMPONENTS ---
def display_prediction_block(predicted_time_str, avg_speed_kmh, inputs, map_data, segment_data, weather_data, predicted_rank_placeholder):
    with st.container(border=True):
        pred_col, stats_col = st.columns([2,1.5])
        with pred_col:
            st.markdown(f"<h3 style='text-align: center;'>Predicted Time</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{predicted_time_str}</h1>", unsafe_allow_html=True)
            st.metric("Average Speed", f"{avg_speed_kmh:.1f} km/h")
            st.info(f"Estimate for an average power of {inputs['power']} W.")
        with stats_col:
            avg_grade = (map_data['elevation'].iloc[-1] - map_data['elevation'].iloc[0]) / segment_data['distance'] * 100 if segment_data['distance'] > 0 else 0
            st.metric("Distance", f"{segment_data['distance']/1000:.2f} km")
            st.metric("Elevation Gain", f"{segment_data['elevation_gain']:.0f} m")
            st.metric("Avg. Grade", f"{avg_grade:.1f}%")
            predicted_rank_placeholder.metric("Predicted Rank", "~", help="Estimated rank based on the public leaderboard.")

        st.markdown("---")
        weather_cols = st.columns(3)
        segment_bearing = calculate_bearing(segment_data['start_latlng'][0], segment_data['start_latlng'][1], map_data['lat'].iloc[-1], map_data['lon'].iloc[-1])
        wind_desc = get_wind_description(segment_bearing, weather_data['wind_direction'])
        wind_cardinal = degrees_to_cardinal(weather_data['wind_direction'])
        weather_cols[0].metric("Temperature", f"{weather_data['temperature']}°C")
        weather_cols[1].metric("Wind Direction", f"{wind_cardinal} - {wind_desc}")
        weather_cols[2].metric("Wind Speed", f"{weather_data['wind_speed']:.1f} km/h")

def display_3d_map(map_data, line_segments_df):
    if "MAPBOX_API_KEY" not in st.secrets:
        st.error("Mapbox API key not found.")
        return
    view_state = pdk.ViewState(latitude=map_data["lat"].mean(), longitude=map_data["lon"].mean(), zoom=12, pitch=60)
    line_layer = pdk.Layer("LineLayer", data=line_segments_df, get_source_position="[lon, lat, elevation]", get_target_position="[lon_next, lat_next, elevation_next]", get_color="color", get_width=5)
    vertical_line_layer = pdk.Layer("LineLayer", data=map_data, get_source_position="[lon, lat, 0]", get_target_position="[lon, lat, elevation]", get_color="color", get_width=1)
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/dark-v10", layers=[line_layer, vertical_line_layer], initial_view_state=view_state))

def display_pacing_chart(map_data, line_segments_df, variable_power):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=map_data['cumulative_distance'] / 1000, y=map_data['smoothed_elevation'], name="Elevation", fill='tozeroy', line=dict(color='grey')), secondary_y=False)
    fig.add_trace(go.Scatter(x=line_segments_df['cumulative_distance'] / 1000, y=variable_power, name="Power Plan", line=dict(color='red')), secondary_y=True)
    fig.update_layout(title_text="Pacing Plan vs. Elevation Profile")
    st.plotly_chart(fig, use_container_width=True)

def display_leaderboard(segment_id, predicted_seconds, athlete_name, segment_data, map_data, line_segments_df, weather_data, inputs, predicted_rank_placeholder):
    st.subheader("🏆 Segment Leaderboard Comparison")
    leaderboard_df = None
    try:
        response = requests.get(f"https://www.strava.com/segments/{segment_id}")
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        if table:
            raw_df = pd.read_html(table.prettify())[0]
            if raw_df.shape[1] >= 4:
                leaderboard_df = raw_df.iloc[:, [0, 2, 3, -1]].copy()
                leaderboard_df.columns = ['Rank', 'Athlete', 'Date', 'Time']
                leaderboard_df['Time (s)'] = leaderboard_df['Time'].apply(parse_time_to_seconds)
                leaderboard_df.dropna(subset=['Time (s)'], inplace=True)
                leaderboard_df['Time (s)'] = leaderboard_df['Time (s)'].astype(int)
            else:
                leaderboard_df = None
    except Exception:
        leaderboard_df = None

    if leaderboard_df is not None and not leaderboard_df.empty:
        predicted_rank = leaderboard_df[leaderboard_df['Time (s)'] < predicted_seconds].shape[0] + 1
        predicted_rank_placeholder.metric("Predicted Rank", f"~{predicted_rank}", help="Estimated rank based on the public leaderboard.")

        if predicted_rank <= 10:
            user_effort = pd.DataFrame([{"Rank": "★", "Athlete": f"{athlete_name} (Your Prediction)", "Time": f"{predicted_seconds // 60}:{predicted_seconds % 60:02d}", "Time (s)": predicted_seconds, "Date": "Forecast"}])
            display_df = pd.concat([leaderboard_df, user_effort]).sort_values(by="Time (s)").head(10)
        else:
            display_df = leaderboard_df.head(10)

        display_df = display_df.drop(columns=['Time (s)'], errors='ignore')
        display_df.reset_index(drop=True, inplace=True)
        display_df['Rank'] = display_df.index + 1
        
        def highlight_user(row):
            if "(Your Prediction)" in str(row['Athlete']):
                return ['background-color: #FF4B4B; color: white'] * len(row)
            return [''] * len(row)

        st.dataframe(display_df[['Rank', 'Athlete', 'Time', 'Date']].style.apply(highlight_user, axis=1), use_container_width=True, hide_index=True)

        if predicted_rank > 10 and len(leaderboard_df) >= 10:
            time_to_beat = leaderboard_df.iloc[9]['Time (s)']
            power_for_top_10 = find_power_for_target_time(time_to_beat, segment_data, map_data, line_segments_df, weather_data, inputs)
            if power_for_top_10:
                power_diff = power_for_top_10 - inputs['power']
                st.warning(f"🎯 To break into the Top 10, you would need to hold an average of **{power_for_top_10} W** (+{power_diff} W).")
    else:
        st.info("Could not display leaderboard at this time.")
