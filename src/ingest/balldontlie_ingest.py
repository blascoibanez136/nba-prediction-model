"""
Ingest historical NBA games from the balldontlie API.

Docs: https://api.balldontlie.io
We use /v1/games with:
  - seasons[]: season year (e.g. 2019 for 2019-20)
  - per_page: page size (max 100)
  - page: pagination

Auth:
  If BALLDONTLIE_API_KEY is set, we send it as an Authorization header.
  (This matches the current balldontlie paid plan header format.)
"""

from __future__ import annotations

import os
from typing import List, Dict

import requests
import pandas as pd

BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = os.getenv("BALLDONTLIE_API_KEY")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if API_KEY:
        # balldontlie uses Authorization header for API keys
        headers["Authorization"] = API_KEY
    return headers


def _fetch_games_for_season(season: int) -> pd.DataFrame:
    """
    Fetch all NBA games for a single season from balldontlie.

    Returns DataFrame with:
        game_date, season, home_team, away_team, home_score, away_score
    """
    all_rows: List[Dict] = []
    page = 1
    per_page = 100

    while True:
        params = {
            "seasons[]": season,
            "per_page": per_page,
            "page": page,
        }
        resp = requests.get(f"{BASE_URL}/games", headers=_headers(), params=params)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("data", [])

        if not games:
            break

        for g in games:
            # balldontlie v1 games structure
            game_date = g.get("date", "")[:10]  # 'YYYY-MM-DDT...' -> 'YYYY-MM-DD'
            season_val = g.get("season", season)

            home_team = g.get("home_team", {}) or {}
            away_team = g.get("visitor_team", {}) or {}

            home_name = home_team.get("full_name") or home_team.get("name")
            away_name = away_team.get("full_name") or away_team.get("name")

            home_score = g.get("home_team_score")
            away_score = g.get("visitor_team_score")

            # Skip incomplete or malformed games
            if (
                home_name is None
                or away_name is None
                or home_score is None
                or away_score is None
            ):
                continue

            all_rows.append(
                {
                    "game_date": game_date,
                    "season": int(season_val),
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                }
            )

        meta = data.get("meta", {})
        total_pages = meta.get("total_pages", page)
        if page >= total_pages:
            break
        page += 1

    return pd.DataFrame(all_rows)


def fetch_games_for_seasons(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Fetch games for all seasons in [start_season, end_season] inclusive.

    Returns DataFrame with:
        game_date, season, home_team, away_team, home_score, away_score
    """
    frames: List[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        print(f"[balldontlie_ingest] Fetching season {season}...")
        df_season = _fetch_games_for_season(season)
        print(f"[balldontlie_ingest] Season {season}: {len(df_season)} games")
        if not df_season.empty:
            frames.append(df_season)

    if not frames:
        return pd.DataFrame(
            columns=["game_date", "season", "home_team", "away_team", "home_score", "away_score"]
        )

    out = pd.concat(frames, ignore_index=True)
    print(f"[balldontlie_ingest] Total games: {len(out)}")
    return out
