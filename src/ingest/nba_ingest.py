import os
from datetime import datetime
import requests

API_KEY = os.getenv("RAPIDAPI_KEY")
BASE_URL = "https://v1.basketball.api-sports.io"

# NBA league ID in API-Sports
NBA_LEAGUE_ID = 12


def _headers():
    if not API_KEY:
        raise RuntimeError(
            "RAPIDAPI_KEY environment variable is not set. "
            "Set it to your API-Sports key."
        )
    return {"x-apisports-key": API_KEY}


def _season_from_date(date_str: str) -> int:
    """
    API-Sports uses the start year of the season as 'season'.
    Example:
      - 2019-10-22 (2019-20 season) -> 2019
      - 2020-03-10 (still 2019-20)   -> 2019
      - 2024-11-01 (2024-25)        -> 2024
    """
    dt = datetime.fromisoformat(date_str)
    return dt.year if dt.month >= 7 else dt.year - 1


def get_games_by_date(date_str: str):
    """All basketball games for the given date (any league)."""
    season = _season_from_date(date_str)
    url = f"{BASE_URL}/games"
    params = {"date": date_str, "season": season}
    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def get_nba_games_by_date(date_str: str):
    """NBA-only games for the given date, using season derived from date."""
    season = _season_from_date(date_str)
    url = f"{BASE_URL}/games"
    params = {
        "date": date_str,
        "league": NBA_LEAGUE_ID,
        "season": season,
    }
    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()

