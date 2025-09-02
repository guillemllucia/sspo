import requests
import streamlit as st
from urllib.parse import urlencode
import time

class StravaAuth:
    def __init__(self):
        self.client_id = st.secrets["STRAVA_CLIENT_ID"]
        self.client_secret = st.secrets["STRAVA_CLIENT_SECRET"]
        self.redirect_uri = st.secrets.get("STRAVA_REDIRECT_URI", "http://localhost:8501")
        self.auth_url = "https://www.strava.com/oauth/authorize"
        self.token_url = "https://www.strava.com/oauth/token"
        self.deauthorize_url = "https://www.strava.com/oauth/deauthorize"

    def get_authorization_url(self):
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "approval_prompt": "force",
            "scope": "read_all,activity:read_all",
        }
        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, authorization_code):
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": authorization_code,
            "grant_type": "authorization_code",
        }
        response = requests.post(self.token_url, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting access token: {response.text}")
            return None

    def refresh_token(self, refresh_token):
        """Uses the refresh token to get a new access token."""
        st.info("Your session has expired. Refreshing your access token...")
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        response = requests.post(self.token_url, data=data)
        if response.status_code == 200:
            st.success("Token refreshed successfully!")
            time.sleep(1) # Give a moment for the user to see the message
            return response.json()
        else:
            st.error("Could not refresh token. Please log in again.")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            return None

    def deauthorize(self, access_token):
        """Revokes the application's access on Strava's side."""
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.post(self.deauthorize_url, headers=headers)
        return response.status_code == 200
