# src/ingest/nba_ingest.py

import os
import requests

API_KEY = os.getenv("RAPIDAPI_KEY")  # we keep your existing env var name
BASE_URL = "https://v1.basketball.api-sports.io"


def _headers():
    # API-Sports native header
    return {
        "x-apisports-key": API_KEY,
    }


def get_games_by_date(date_str: str):
    """
    Fetch games for a given date from API-Sports Basketball.
    Example: '2025-11-10' or '2025-01-15'
    """
    url = f"{BASE_URL}/games"
    params = {"date": date_str}
    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()

