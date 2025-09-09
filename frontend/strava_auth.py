import requests
import streamlit as st
from urllib.parse import urlencode
<<<<<<< HEAD
import time
=======

>>>>>>> 786d3e58bdd990b518fb765be958f8afb1f9e41a

class StravaAuth:
    def __init__(self):
        self.client_id = st.secrets["STRAVA_CLIENT_ID"]
        self.client_secret = st.secrets["STRAVA_CLIENT_SECRET"]
<<<<<<< HEAD
        self.redirect_uri = st.secrets.get("STRAVA_REDIRECT_URI", "http://localhost:8501")
        self.auth_url = "https://www.strava.com/oauth/authorize"
        self.token_url = "https://www.strava.com/oauth/token"
        self.deauthorize_url = "https://www.strava.com/oauth/deauthorize"
=======
        self.redirect_uri = st.secrets.get(
            "STRAVA_REDIRECT_URI", "http://localhost:8501"
        )
        self.auth_url = "https://www.strava.com/oauth/authorize"
        self.token_url = "https://www.strava.com/oauth/token"
        self.api_url = "https://www.strava.com/api/v3"
>>>>>>> 786d3e58bdd990b518fb765be958f8afb1f9e41a

    def get_authorization_url(self):
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "approval_prompt": "force",
<<<<<<< HEAD
            "scope": "read_all,activity:read_all",
=======
            "scope": "read,activity:read_all",
>>>>>>> 786d3e58bdd990b518fb765be958f8afb1f9e41a
        }
        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, authorization_code):
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": authorization_code,
            "grant_type": "authorization_code",
        }
<<<<<<< HEAD
        response = requests.post(self.token_url, data=data)
=======

        response = requests.post(self.token_url, data=data)

>>>>>>> 786d3e58bdd990b518fb765be958f8afb1f9e41a
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting access token: {response.text}")
            return None

<<<<<<< HEAD
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
=======
    def get_athlete_info(self, access_token):
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.api_url}/athlete", headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting athlete info: {response.text}")
            return None

    def get_activity_stats(self, access_token, athlete_id):
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{self.api_url}/athletes/{athlete_id}/stats", headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting activity stats: {response.text}")
            return None

    def get_activities(self, access_token, per_page=30, page=1):
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"per_page": per_page, "page": page}
        response = requests.get(
            f"{self.api_url}/athlete/activities", headers=headers, params=params
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting activities: {response.text}")
            return None
>>>>>>> 786d3e58bdd990b518fb765be958f8afb1f9e41a
