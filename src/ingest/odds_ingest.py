"""
Odds ingestion for NBA using The Odds API.

This module exposes a single function:

    get_nba_odds() -> list[dict]

which is used by src.ingest.odds_snapshots.save_snapshot() to
create raw odds snapshots (with nested 'bookmakers' data) that
are later flattened and used for dispersion/consensus calculations.

Environment:
    ODDS_API_KEY  - your The Odds API key
"""

from __future__ import annotations

import os
from typing import Any, List

import requests

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def get_nba_odds(
    *,
    markets: str = "h2h,spreads,totals",
    regions: str = "us",
    odds_format: str = "american",
) -> List[dict[str, Any]]:
    """
    Fetch current NBA odds from The Odds API.

    Parameters
    ----------
    markets : str, optional
        Comma-separated list of markets to request. Defaults to
        "h2h,spreads,totals" which is sufficient for spreads + totals.
    regions : str, optional
        Regions to request (The Odds API format). "us" is typical.
    odds_format : str, optional
        "american", "decimal", etc. "american" is standard for US books.

    Returns
    -------
    list[dict]
        List of game JSON objects as returned by The Odds API.
        Each object includes:
            - id
            - home_team
            - away_team
            - commence_time (ISO8601)
            - bookmakers: [ { key/title, markets: [...] }, ... ]
    """
    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is not set. "
            "Set it in your environment / GitHub Actions secrets "
            "to use The Odds API."
        )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }

    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected response from The Odds API (expected list, got {type(data)})"
        )

    return data


if __name__ == "__main__":
    # Simple manual test:
    games = get_nba_odds()
    print(f"Fetched {len(games)} games from The Odds API.")
    if games:
        sample = games[0]
        print("Sample game keys:", list(sample.keys()))
