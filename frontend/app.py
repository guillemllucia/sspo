# app.py
import streamlit as st
from strava_auth import StravaAuth
import base64
from pathlib import Path

# --- CONFIGURATION & SESSION STATE ---
st.set_page_config(page_title="Strava Optimizer Login", page_icon="🚴", layout="wide")

# --- Hide sidebar and header anchors on login page ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "athlete_info" not in st.session_state:
    st.session_state.athlete_info = None

# --- UI FUNCTIONS ---
def set_bg_hack(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;
             background-position: center 70%;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def show_authentication_page(auth):
    set_bg_hack("https://trello-backgrounds.s3.amazonaws.com/SharedBackground/2560x1440/1609b2cc34793439f41f21b944076194/photo-1534787238916-9ba6764efd4f.webp")
    st.markdown("""
        <style>
        .glass-container {
            background: rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            color: white;
        }
        .glass-container h1, .glass-container h2, .glass-container h3, .glass-container p, .glass-container li, .glass-container h4 {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    auth_url = auth.get_authorization_url()

    script_dir = Path(__file__).parent
    button_image_path = script_dir / "btn_strava_connect_with_orange.png"
    if button_image_path.is_file():
        with open(button_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        connect_button_html = f"<a href='{auth_url}' target='_blank'><img src='data:image/png;base64,{image_base64}' alt='Connect with Strava'></a>"
    else:
        connect_button_html = f'<a href="{auth_url}" target="_blank" style="display: block; padding: 0.5em 1em; background-color: #FC4C02; color: white; text-decoration: none; border-radius: 0.5rem; text-align: center; width: 100%; box-sizing: border-box;">🔗 Connect to Strava</a>'
        st.warning(f"Login button image not found at path: {button_image_path}")

    attribution_logo_path = script_dir / "api_logo_cptblWith_strava_horiz_white.png"
    attribution_html = ""
    if attribution_logo_path.is_file():
        with open(attribution_logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        attribution_html = f"<div style='text-align: center; margin-top: 15px; margin-bottom: 15px;'><img src='data:image/png;base64,{logo_base64}' alt='Compatible with Strava' style='width: 200px;'></div>"

    login_html = f"""
    <div class="glass-container">
        <h2 style='text-align: center;'>🚴‍♂️ Strava Pacing Optimizer</h2>
        <h4 style='text-align: center; font-weight: 400;'>Your personal pacing strategist for Strava.</h4>
        <hr>
        <div style="text-align: center;">
            {connect_button_html}
        </div>
            <h>
            {attribution_html}

    </div>
    """

    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.markdown(login_html, unsafe_allow_html=True)

def show_main_app_redirect():
    """ A simple page to show when a user is already logged in, redirecting them to the main app."""
    st.title("Welcome Back!")
    st.success("You are already logged in.")
    st.page_link("pages/0_Main_Dashboard.py", label="Go to Your Dashboard", icon="🚀")


# --- MAIN ROUTER ---
def main():
    auth = StravaAuth()

    if st.session_state.get("authenticated"):
        st.switch_page("pages/0_Main_Dashboard.py")

    query_params = st.query_params

    if "code" in query_params:
        authorization_code = query_params.get("code")
        with st.spinner("Authenticating with Strava..."):
            token_data = auth.exchange_code_for_token(authorization_code)
            if token_data:
                st.session_state.access_token = token_data["access_token"]
                st.session_state.refresh_token = token_data.get("refresh_token")
                st.session_state.expires_at = token_data.get("expires_at")
                st.session_state.athlete_info = token_data["athlete"]
                st.session_state.authenticated = True
                st.switch_page("pages/0_Main_Dashboard.py")
            else:
                st.error("Authentication failed. Please try again.")
    else:
        show_authentication_page(auth)

if __name__ == "__main__":
    main()
