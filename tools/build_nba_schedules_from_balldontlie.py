# tools/build_nba_schedules_from_balldontlie.py
import os
import requests
import pandas as pd
from pathlib import Path

API_URL = "https://api.balldontlie.io/v1/games"
API_KEY = os.getenv("BALLDONTLIE_API_KEY")


def fetch_season(season_year: int) -> pd.DataFrame:
    """
    Fetch ALL games for a season using balldontlie's pagination.
    Returns a DataFrame with columns: game_date, home_team, away_team
    """
    if not API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY environment variable not set")

    headers = {"Authorization": API_KEY}

    all_rows = []
    page = 1

    print(f"Fetching season {season_year}...")

    while True:
        params = {
            "seasons[]": season_year,
            "page": page,
            "per_page": 100,   # max allowed
        }

        r = requests.get(API_URL, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

        if len(data["data"]) == 0:
            break  # no more pages

        for g in data["data"]:
            all_rows.append({
                "game_date": g["date"][:10],  # YYYY-MM-DD
                "home_team": g["home_team"]["full_name"],
                "away_team": g["visitor_team"]["full_name"],
            })

        print(f"  Page {page}: {len(data['data'])} games")

        page += 1

    print(f"Total games fetched for {season_year}: {len(all_rows)}")
    return pd.DataFrame(all_rows)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "schedules"
    out_dir.mkdir(parents=True, exist_ok=True)

    df1 = fetch_season(2023)  # 2023–24 season
    df2 = fetch_season(2024)  # 2024–25 season (if available)

    combined = pd.concat([df1, df2], ignore_index=True)

    out_path = out_dir / "nba_schedule.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined schedule with {len(combined)} games to {out_path}")


if __name__ == "__main__":
    main()

