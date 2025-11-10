import os
import requests

ODDS_API_KEY = os.getenv("ODDS_API_KEY")


def get_nba_odds():
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()
