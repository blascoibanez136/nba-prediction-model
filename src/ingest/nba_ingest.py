import os
import requests

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

BASE_URL = "https://api-nba-v1.p.rapidapi.com"


def _headers():
    return {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com",
    }


def get_games_by_date(date_str: str):
    """
    date_str example: '2024-10-25'
    """
    url = f"{BASE_URL}/games"
    resp = requests.get(url, headers=_headers(), params={"date": date_str})
    resp.raise_for_status()
    return resp.json()
