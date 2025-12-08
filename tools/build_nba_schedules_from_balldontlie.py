#!/usr/bin/env python3
"""
Build NBA schedule CSVs from the Balldontlie API.
Works in GitHub Actions AND Colab (no __file__ required).
"""

import os
import requests
import pandas as pd
from pathlib import Path

API_URL = "https://api.balldontlie.io/v1/games"
API_KEY = os.getenv("BALLDONTLIE_API_KEY")


def fetch_season(season_year: int) -> pd.DataFrame:
    print(f"Fetching season {season_year}…")
    headers = {"Authorization": API_KEY} if API_KEY else {}
    all_games = []

    page = 1
    while True:
        resp = requests.get(
            API_URL,
            params={"seasons[]": season_year, "per_page": 100, "page": page},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        for g in data["data"]:
            all_games.append({
                "game_date": g["date"][:10],
                "home_team": g["home_team"]["full_name"],
                "away_team": g["visitor_team"]["full_name"],
            })

        if page >= data["meta"]["total_pages"]:
            break
        page += 1

    df = pd.DataFrame(all_games)
    print(f" → {len(df)} games fetched.")
    return df


def main():
    if not API_KEY:
        raise ValueError("BALLDONTLIE_API_KEY is missing. Set it in your environment.")

    # Colab-safe repo root
    repo_root = Path.cwd()
    sched_dir = repo_root / "data" / "schedules"
    sched_dir.mkdir(parents=True, exist_ok=True)

    df_2324 = fetch_season(2023)
    df_2425 = fetch_season(2024)

    df_2324.to_csv(sched_dir / "nba_schedule_2023_24.csv", index=False)
    df_2425.to_csv(sched_dir / "nba_schedule_2024_25.csv", index=False)

    df_combined = pd.concat([df_2324, df_2425], ignore_index=True)
    df_combined.to_csv(sched_dir / "nba_schedule.csv", index=False)

    print(f"Saved combined schedule → {sched_dir / 'nba_schedule.csv'}")


if __name__ == "__main__":
    main()
