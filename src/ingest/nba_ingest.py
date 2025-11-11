import os
import requests

API_KEY = os.getenv("RAPIDAPI_KEY")
BASE_URL = "https://v1.basketball.api-sports.io"

def _headers():
    return {"x-apisports-key": API_KEY}

def get_games_by_date(date_str: str):
    url = f"{BASE_URL}/games"
    params = {"date": date_str}
    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()

