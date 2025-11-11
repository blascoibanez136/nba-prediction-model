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


def get_nba_games_by_date(date_str: str):
    """NBA-only convenience wrapper."""
    # 1) get leagues
    leagues_resp = requests.get(f"{BASE_URL}/leagues", headers=_headers())
    leagues_resp.raise_for_status()
    leagues = leagues_resp.json()["response"]

    # 2) find NBA
    nba_leagues = [lg for lg in leagues if "nba" in lg["name"].lower()]
    if not nba_leagues:
        raise RuntimeError("NBA league not found in API-Sports")
    nba = nba_leagues[0]
    league_id = nba["id"]

    # 3) pick latest season
    seasons = [s["season"] for s in nba.get("seasons", [])]
    season = sorted(seasons)[-1]

    # 4) fetch NBA games for that date
    url = f"{BASE_URL}/games"
    params = {
        "date": date_str,
        "league": league_id,
        "season": season,
    }
    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()
