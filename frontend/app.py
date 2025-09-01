# app.py
import streamlit as st
from strava_auth import StravaAuth

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
             background-position: center 20%;
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
        .glass-container h1, .glass-container h2, .glass-container h3, .glass-container p, .glass-container li {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    auth_url = auth.get_authorization_url()

    login_html = f"""
    <div class="glass-container">
        <h1 style='text-align: center;'>🚴‍♂️ Strava Pacing Optimizer</h1>
        <h2 style='text-align: center; font-style: italic; font-weight: 400;'>Your personal pacing strategist for any Strava segment.</h2>
        <hr>
        <h3 style='text-align: center; font-weight: 400;'>🔐 Connect to Strava to begin</h3>
        <a href="{auth_url}" target="_self" style="display: block; padding: 0.5em 1em; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.5rem; text-align: center; width: 100%; box-sizing: border-box;">🔗 Connect to Strava</a>
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
    query_params = st.query_params

    if "code" in query_params and not st.session_state.authenticated:
        authorization_code = query_params.get("code")
        with st.spinner("Authenticating with Strava..."):
            token_data = auth.exchange_code_for_token(authorization_code)
            if token_data:
                st.session_state.access_token = token_data["access_token"]
                st.session_state.athlete_info = token_data["athlete"]
                st.session_state.authenticated = True
                st.switch_page("pages/0_Main_Dashboard.py")
            else:
                st.error("Authentication failed.")

    if st.session_state.authenticated:
        show_main_app_redirect()
    else:
        show_authentication_page(auth)

if __name__ == "__main__":
    main()

