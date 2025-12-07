"""
NBA / basketball ingest from API-Sports.

We use the v1 Basketball API:
    https://v1.basketball.api-sports.io

Auth:
    RAPIDAPI_KEY env var must contain your API-Sports key.
    We send it via the "x-apisports-key" header.

Functions:
    get_games_by_date(date_str)
    get_nba_games_by_date(date_str)

get_nba_games_by_date strategy:
    1) Discover NBA league_id + seasons from /leagues.
    2) Try /games with (league, date) ONLY (no season).
    3) If that yields 0 games, fall back to (league, latest season, date).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any

import requests

API_KEY = os.getenv("RAPIDAPI_KEY")
BASE_URL = "https://v1.basketball.api-sports.io"


def _headers() -> Dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "RAPIDAPI_KEY is not set. "
            "Set it to your API-Sports basketball API key."
        )
    return {"x-apisports-key": API_KEY}


def get_games_by_date(date_str: str, extra_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Generic games-by-date call.

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD' format.
    extra_params : dict, optional
        Additional query parameters (e.g. league, season).

    Returns
    -------
    dict
        Raw JSON response from API-Sports.
    """
    url = f"{BASE_URL}/games"
    params: Dict[str, Any] = {"date": date_str}
    if extra_params:
        params.update(extra_params)

    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def _get_nba_league_and_seasons() -> Tuple[int, List[int]]:
    """
    Discover the NBA league_id and list of seasons from /leagues.

    Returns
    -------
    (league_id, seasons) : (int, list[int])
    """
    url = f"{BASE_URL}/leagues"
    resp = requests.get(url, headers=_headers())
    resp.raise_for_status()
    leagues = resp.json().get("response", [])

    nba_leagues = [lg for lg in leagues if "nba" in (lg.get("name") or "").lower()]
    if not nba_leagues:
        raise RuntimeError("NBA league not found in API-Sports /leagues response")

    nba = nba_leagues[0]
    league_id = nba["id"]
    seasons = sorted({s["season"] for s in nba.get("seasons", []) if "season" in s})

    return league_id, seasons


def get_nba_games_by_date(date_str: str) -> Dict[str, Any]:
    """
    NBA-only convenience wrapper around /games.

    Strategy:
      1) Find NBA league id from /leagues.
      2) First try /games with (league, date) ONLY (no season).
      3) If that yields 0 games, fall back to (league, latest season, date).

    This is more robust across pre-season / regular-season / future schedules,
    where the API's notion of "season" might not align neatly with the date.
    """
    league_id, seasons = _get_nba_league_and_seasons()
    print(f"[nba_ingest] Using NBA league_id={league_id}, seasons={seasons}")

    # --- Attempt 1: league + date (no season) ---
    data = get_games_by_date(date_str, extra_params={"league": league_id})
    games = data.get("response", [])
    print(f"[nba_ingest] Attempt 1 (league+date) for {date_str}: {len(games)} games")

    if games:
        return data

    # --- Attempt 2: league + latest season + date (legacy behavior) ---
    if seasons:
        latest_season = seasons[-1]
        print(
            f"[nba_ingest] No games found in attempt 1; "
            f"trying league+season+date with season={latest_season}"
        )
        data2 = get_games_by_date(
            date_str,
            extra_params={"league": league_id, "season": latest_season},
        )
        games2 = data2.get("response", [])
        print(
            f"[nba_ingest] Attempt 2 (league+season+date) "
            f"for {date_str}, season={latest_season}: {len(games2)} games"
        )
        return data2

    # If no seasons info or both attempts failed, return the first (likely empty) result
    print(
        f"[nba_ingest] No seasons metadata or no games found for {date_str} "
        f"even after fallback. Returning empty response."
    )
    return data
