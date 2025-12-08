"""
tools/build_nba_schedules_from_balldontlie.py

Builds season-long NBA schedule CSVs using the balldontlie API.

Outputs:
    data/schedules/nba_schedule_2023_24.csv
    data/schedules/nba_schedule_2024_25.csv
    data/schedules/nba_schedule.csv          (combined)

Requirements:
    - Environment variable BALLDONTLIE_API_KEY must be set.
    - requests and pandas must be installed (already in your project).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests

API_KEY = os.getenv("BALLDONTLIE_API_KEY")
BASE_URL = "https://api.balldontlie.io/v1/games"


def _check_api_key() -> None:
    if not API_KEY:
        raise RuntimeError(
            "BALLDONTLIE_API_KEY is not set. "
            "Export it in your environment before running this script."
        )


def fetch_season_games(season: int) -> pd.DataFrame:
    """
    Fetch all regular-season games for a given season from balldontlie.

    balldontlie treats `season=2023` as the 2023-24 NBA season, etc.

    Returns a DataFrame with:
        game_date, home_team, away_team
    """
    headers = {"Authorization": API_KEY}
    per_page = 100
    page = 1
    rows: List[Dict[str, Any]] = []

    while True:
        params = {
            "seasons[]": season,
            "per_page": per_page,
            "page": page,
        }
        resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        games = payload.get("data", [])

        if not games:
            break

        for g in games:
            # Example structure:
            # {
            #   "id": ...,
            #   "date": "2024-02-05T00:00:00.000Z",
            #   "home_team": {...},
            #   "visitor_team": {...},
            #   ...
            # }
            d = g.get("date") or ""
            game_date = d[:10] if d else None

            home = g.get("home_team", {}) or {}
            away = g.get("visitor_team", {}) or {}
            home_name = home.get("full_name") or home.get("name")
            away_name = away.get("full_name") or away.get("name")

            if not (game_date and home_name and away_name):
                continue

            rows.append(
                {
                    "game_date": game_date,
                    "home_team": home_name,
                    "away_team": away_name,
                }
            )

        meta = payload.get("meta", {})
        next_page = meta.get("next_page")
        if not next_page:
            break

        page = next_page
        # Be nice to the API
        time.sleep(0.25)

    df = pd.DataFrame(rows, columns=["game_date", "home_team", "away_team"])
    df = df.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def main() -> None:
    _check_api_key()

    repo_root = Path(__file__).resolve().parents[1]
    sched_dir = repo_root / "data" / "schedules"
    sched_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching 2023-24 season (season=2023)...")
    df_2324 = fetch_season_games(2023)
    out_2324 = sched_dir / "nba_schedule_2023_24.csv"
    df_2324.to_csv(out_2324, index=False)
    print(f"Saved {len(df_2324)} games to {out_2324}")

    print("Fetching 2024-25 season (season=2024)...")
    df_2425 = fetch_season_games(2024)
    out_2425 = sched_dir / "nba_schedule_2024_25.csv"
    df_2425.to_csv(out_2425, index=False)
    print(f"Saved {len(df_2425)} games to {out_2425}")

    # Combined schedule (useful as a generic fallback)
    combined = pd.concat([df_2324, df_2425], ignore_index=True)
    combined = combined.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    out_all = sched_dir / "nba_schedule.csv"
    combined.to_csv(out_all, index=False)
    print(f"Saved combined schedule ({len(combined)} games) to {out_all}")


if __name__ == "__main__":
    main()
