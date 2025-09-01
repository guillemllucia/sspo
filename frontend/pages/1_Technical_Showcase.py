import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime
from sspo_helpers import * # Import all our shared functions

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Technical Showcase")

# --- Hide sidebar navigation and header anchor links ---
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE CHECK & TOKEN REFRESH ---
if not st.session_state.get("authenticated"):
    st.error("Please log in to view this page.")
    st.link_button("Go to Login", "/")
    st.stop()

if 'expires_at' in st.session_state and datetime.now().timestamp() > st.session_state.expires_at:
    from strava_auth import StravaAuth
    auth = StravaAuth()
    token_data = auth.refresh_token(st.session_state.refresh_token)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data["refresh_token"]
        st.session_state.expires_at = token_data["expires_at"]
        st.rerun()
    else:
        st.stop()

# --- Initialize session state for demo steps ---
if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 1
if 'demo_segment_url' not in st.session_state:
    st.session_state.demo_segment_url = "https://www.strava.com/segments/13260861"
if 'demo_weight' not in st.session_state:
    st.session_state.demo_weight = 75
if 'demo_power' not in st.session_state:
    st.session_state.demo_power = 250

# --- DEMO CONTROLS ---
st.sidebar.header("Technical Showcase")
st.sidebar.info("A step-by-step look at how the Optimizer works.")
if st.sidebar.button("Exit Showcase"):
    st.session_state.demo_step = 1 # Reset step for next time
    st.switch_page("pages/0_Main_Dashboard.py")

# --- SHOWCASE CONTENT ---
st.title("🚀 Technical Showcase")
st.markdown("---")

# --- Navigation Buttons ---
col_prev, col_next, _ = st.columns([1,1,8])
with col_prev:
    if st.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.demo_step <= 1)):
        st.session_state.demo_step -= 1
        st.rerun()
with col_next:
    if st.session_state.demo_step < 5:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.demo_step += 1
            st.rerun()
    else:
        if st.button("Finish Showcase 🎉", use_container_width=True):
            st.session_state.demo_step = 1
            st.switch_page("pages/0_Main_Dashboard.py")

# --- Step-by-step logic ---
if st.session_state.demo_step == 1:
    st.header("Step 1: Data Collection & User Input")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("""
        The process begins by gathering three key types of data:
        1.  **User Input:** The target segment, athlete's weight, and desired power are collected.
        2.  **Segment Data:** The app calls the Strava API for the segment's core details.
        3.  **Environmental Data:** The app calls the Open-Meteo API for a precise weather and elevation forecast.
        """)
        st.success("Enter your own parameters on the right, or use the defaults. Then, click 'Next' to see how this raw data is sent to our prediction model.")
    with col2:
        st.subheader("Interactive Inputs")
        with st.container(border=True):
            st.session_state.demo_segment_url = st.text_input("Strava Segment URL or ID:", value=st.session_state.demo_segment_url)
            st.session_state.demo_weight = st.number_input("Your Weight (kg):", value=st.session_state.demo_weight)
            st.session_state.demo_power = st.number_input("Target Power (Watts):", value=st.session_state.demo_power)

elif st.session_state.demo_step >= 2:
    segment_id = st.session_state.demo_segment_url.split('/')[-1]

    with st.spinner("Fetching live data for showcase..."):
        segment_data = get_segment_data(segment_id, st.session_state.access_token)

        if segment_data is None or segment_data['map_data'].empty:
            st.error(f"Could not fetch data for segment {segment_id}. Please check the URL and your access token.")
            st.stop()

        map_data = get_elevation_data(segment_data.get("map_data").copy())
        if map_data is None:
            st.error("Could not fetch elevation data for this segment.")
            st.stop()

    map_data['lat_next'] = map_data['lat'].shift(-1); map_data['lon_next'] = map_data['lon'].shift(-1)
    map_data['distance_segment'] = map_data.apply(lambda r: haversine_np(r['lon'], r['lat'], r['lon_next'], r['lat_next']) if pd.notna(r['lon_next']) else 0, axis=1)
    map_data['cumulative_distance'] = map_data['distance_segment'].cumsum()
    map_data['smoothed_elevation'] = map_data['elevation'].rolling(window=25, center=True, min_periods=1).mean()
    map_data['elevation_next'] = map_data['smoothed_elevation'].shift(-1)
    line_segments_df = map_data.dropna(subset=['lon_next', 'lat_next']).copy()
    line_segments_df['elevation_change'] = line_segments_df['elevation_next'] - line_segments_df['smoothed_elevation']
    line_segments_df['gradient'] = line_segments_df.apply(lambda r: (r['elevation_change'] / r['distance_segment']) * 100 if r['distance_segment'] > 0 else 0, axis=1)
    line_segments_df['color'] = line_segments_df['gradient'].apply(get_color_from_gradient)

    predicted_seconds = int(936 * (250 / st.session_state.demo_power))
    predicted_time_str = f"{predicted_seconds // 60}:{predicted_seconds % 60:02d}"
    avg_speed_kmh = (segment_data['distance'] / predicted_seconds) * 3.6 if predicted_seconds > 0 else 0

    if st.session_state.demo_step == 2:
        st.header("Step 2: The Prediction")
        st.info("The collected data is sent to our backend API, which returns a time prediction.")
        display_prediction_block(predicted_time_str, avg_speed_kmh, {"power": st.session_state.demo_power}, map_data, segment_data, {"temperature": 22, "wind_direction": 270, "wind_speed": 10.5}, st.empty())

    elif st.session_state.demo_step == 3:
        st.header("Step 3: 3D Visualization")
        st.info("The app generates an interactive 3D map of the segment's topography.")
        display_3d_map(map_data, line_segments_df)

    elif st.session_state.demo_step == 4:
        st.header("Step 4: The Pacing Plan")
        st.info("An elevation profile chart shows the user how to vary their power output.")
        variable_power = (st.session_state.demo_power + (line_segments_df['gradient'] * 10)).clip(lower=0)
        display_pacing_chart(map_data, line_segments_df, variable_power)

    elif st.session_state.demo_step == 5:
        st.header("Step 5: The Competitive Edge")
        st.info("The app shows the user where their predicted time would place them on the leaderboard.")
        display_leaderboard(segment_id, predicted_seconds, "Demo User", segment_data, map_data, line_segments_df, {}, {"power": st.session_state.demo_power}, st.empty())
