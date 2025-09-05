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
        [data-testid="stSidebar"] {
            width: 250px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    steps = {
        1: "What is Strava?",
        2: "Pacing is Complex",
        3: "Data & Model",
        4: "Modeling & Visualization",
        5: "Performance & Limitations"
    }

    if st.session_state.demo_step < 3:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.demo_step += 1
            st.rerun()
    else:
        if st.button("🚀 Product Demo", use_container_width=True, type="primary"):
            st.session_state.demo_step = 1
            st.switch_page("pages/0_Main_Dashboard.py")

    if st.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.demo_step <= 1)):
        st.session_state.demo_step -= 1
        st.rerun()

    st.markdown("---")

    current_step_title = steps.get(st.session_state.demo_step, "")
    st.markdown(f'<h1 style="color: #FF4B4B;">{current_step_title}</h1>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    if st.button("Exit Showcase", type='secondary', use_container_width=True):
        st.session_state.demo_step = 1 # Reset step for next time
        st.switch_page("pages/0_Main_Dashboard.py")

# --- Step-by-step logic ---
if st.session_state.demo_step == 1:
    col1, col2 = st.columns([1.75,1])
    with col1:
        st.info("""
        ### 🏆 Strava is the #1 social network for athletes, supporting many sports.
        ### 🚴 One of Strava's most popular sports is cycling.
        ### 👑 In the cycling world, users compete for the best times on Segments.
        ### ⛰️ Achieving a Top 10 in a leaderboard is a major goal for many cyclists.
        ######
        """)
    with col2:
        st.image("https://codaio.imgix.net/packs/10562/unversioned/assets/COVER/726b965ab3788b6d569a33a466ef1804f948cff6f3d76c09a91cb58f3f483fb96310a5f3f784fc7203f63b4dafc554e9f5769555f9281ce83e55754d4b28c174fa969ee6565a6258652e908cecd906bebf073c347364fc193804a6e0ab5cf0e8081b4f10?fit=crop&ar=1.91%3A1&fm=jpg")
    st.image("https://images2.giant-bicycles.com/b_white%2Cc_pad%2Ch_600%2Cq_80%2Cw_1920/jt7khbh9gwvgdixi9o9g/Banner_lingo.jpg")


elif st.session_state.demo_step == 2:
    col1, col2 = st.columns([1.15,1])
    with col1:
        st.info("""
        ### 📈 Athletes want to improve, but don't know how the conditions will affect their effort.
        ### 🤔 A record breaking ride is a difficult balance of fitness, terrain, and weather.
        ### 😰 Athletes don't want to waste an effort in non-optimal conditions.
        ### ❓ The Question: How can a cyclist know the conditions from the get-go?
        ######
        """)
        st.image("https://i.postimg.cc/43cHPW9T/Screenshot-2025-09-05-142414.png", use_container_width=True)
    with col2:
        st.image("https://i.postimg.cc/sXSPBVc7/Screenshot-2025-09-05-142235.png", use_container_width=True)

elif st.session_state.demo_step == 3:
    col1, col2 = st.columns([1.08,1])
    with col1:
        st.image("https://cdn.mos.cms.futurecdn.net/Um7RXXdVVRWqxj76kY3GoQ.jpg")
    with col2:
        st.info("""
        ### 📲 Data originates from the past rides of cyclists at all skill levels.
        ### 📊 The dataset includes key metrics such as location, ride time, and effort.
        ### ⚙️ This collection provides 86,000 real segments for processing.
        ### 🤝 The segment data then merges with meteorological data to account for weather conditions.
        ### ☁️ This final dataset trains an XGBoost Regressor model to predict ride times.
        #####
        """)
    col1, col2 = st.columns([1.06,1])
    with col1:
        st.image("https://i.postimg.cc/j5S0tSrq/Screenshot-2025-09-05-163557rtrt.png", use_container_width=True)
    with col2:
        st.image("https://i.postimg.cc/qvsNSjLc/Screenshot-2025-09-05-161448.png", use_container_width=True)
