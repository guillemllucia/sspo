import streamlit as st
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Technical Showcase")

# --- Initialize session state ---
if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 1

# --- Hide sidebar navigation and header anchor links ---
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
        .showcase-text {
            text-align: justify;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    steps = {
        1: "What is Strava?",
        2: "Pacing is Complex",
        3: "Data Integration",
        4: "Modeling & Visualization",
        5: "Performance & Limitations"
    }

    current_step_title = steps.get(st.session_state.demo_step, "")
    st.markdown(f'<h1 style="color: #FF4B4B;">{current_step_title}</h1>', unsafe_allow_html=True)

    st.markdown("---")

    col_prev, col_next = st.columns(2)
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
            if st.button("🚀 Product Demo", use_container_width=True, type="primary"):
                st.session_state.demo_step = 1
                st.switch_page("pages/0_Main_Dashboard.py")

    if st.button("Exit Showcase"):
        st.session_state.demo_step = 1 # Reset step for next time
        st.switch_page("pages/0_Main_Dashboard.py")

# --- Step-by-step logic ---
if st.session_state.demo_step == 1:
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("""
        * ### 🏆 Strava is the #1 social network for athletes, supporting many sports.
        * ### 🚴 Our project focuses specifically on the world of **cycling**.
        * ### 👑 In the cycling world, users compete for the best times on **Segments**.
        * ### ⛰️ Achieving a **King of the Mountain (KOM)** is a major goal for many cyclists.
        """)
    with col2:
        st.image("https://i.imgur.com/example_leaderboard.png", caption="The coveted Strava Leaderboard, where legends are made.")

elif st.session_state.demo_step == 2:
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("""
        * ### 📈 Athletes want to improve, but don't know the exact effort required.
        * ### 🤔 Perfect pacing is a difficult balance of fitness, terrain, and weather.
        * ### 😰 Athletes don't want to risk a maximal effort just to test a pacing strategy.
        * ### ❓ **The Question:** How can a cyclist know the right effort from the get-go?
        """)
    with col2:
        st.image("https://i.imgur.com/example_elevation.png", caption="The constantly changing gradient of a segment makes manual pacing a significant challenge.")

elif st.session_state.demo_step == 3:
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("""
        * ### 📲 **Frontend:** A clean, interactive user interface built with Streamlit.
        * ### ☁️ **Backend:** A powerful XGBoost model served via a FastAPI on Google Cloud Run.
        * ### 🌐 **Live Data:** We enrich user inputs by calling two external APIs:
            * **Strava API for official segment data.**
            * **Open-Meteo API for hyper-local weather & elevation.**
        """)
    with col2:
        st.image("https://i.imgur.com/example_architecture.png", caption="The application's high-level architecture.")

elif st.session_state.demo_step == 4:
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("""
        * ### 🧠 **The Model:** We trained an **XGBoost Regressor** for its world-class performance.
        * ### 📊 **Training Data:** The model learned from thousands of real-world cycling efforts.
        * ### ✨ **The Magic:** It takes 13 key features and outputs a single, powerful number: the **predicted time in seconds**.
        * ### 🗺️ **Visualization:** The app then renders a full dashboard with a 3D map, elevation profile, and more.
        """)
    with col2:
         st.image("https://i.imgur.com/example_features.png", caption="The 13 features used by the XGBoost model to make its prediction.")

elif st.session_state.demo_step == 5:
    col1, col2 = st.columns([1,1])
    with col1:
        st.info("""
        * ### ✅ **Performance:** The model is highly accurate, with a competitive Root Mean Squared Error (RMSE).
        * ### 🚧 **Limitations:** It can't account for real-world variables like traffic, road surface, or a cyclist's daily form.
        * ### 💡 **Future Work:** Performance can be further improved with more data and extensive hyperparameter tuning.
        * ### 🏆 **The Result:** A powerful tool that transforms raw data into a clear, actionable, and highly motivational pacing plan.
        """)
    with col2:
        st.image("https://i.imgur.com/example_final_dashboard.png", caption="The final dashboard, combining prediction, visualization, and competitive analysis.")
