# tools/backfill_schedule_dates.py

"""
Backfill specific dates into data/schedules/nba_schedule.csv
using the Balldontlie API.

Usage (in Colab or locally):

    export BALLDONTLIE_API_KEY="your_key"
    export SCHEDULE_DATES="2024-02-05,2024-02-10"

    python tools/backfill_schedule_dates.py

If data/schedules/nba_schedule.csv exists, new rows are appended
and de-duplicated. Otherwise, the file is created.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests

API_URL = "https://api.balldontlie.io/v1/games"
API_KEY = os.getenv("BALLDONTLIE_API_KEY")
DATES_RAW = os.getenv("SCHEDULE_DATES", "").strip()


def _parse_dates() -> List[str]:
    if not DATES_RAW:
        raise RuntimeError(
            "SCHEDULE_DATES environment variable is empty. "
            "Example: SCHEDULE_DATES='2024-02-05,2024-02-10'"
        )
    out: List[str] = []
    for part in DATES_RAW.split(","):
        d = part.strip()
        if d:
            out.append(d)
    if not out:
        raise RuntimeError("No valid dates parsed from SCHEDULE_DATES.")
    return out


def fetch_games_for_date(date_str: str) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY environment variable not set.")

    print(f"[backfill] Fetching games for {date_str}...")
    headers = {"Authorization": API_KEY}
    params = {"dates[]": date_str, "per_page": 100}
    resp = requests.get(API_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    games = payload.get("data", [])

    rows: List[Dict[str, Any]] = []
    for g in games:
        rows.append(
            {
                "game_date": date_str,
                "home_team": g["home_team"]["full_name"],
                "away_team": g["visitor_team"]["full_name"],
            }
        )

    df = pd.DataFrame(rows, columns=["game_date", "home_team", "away_team"])
    print(f"[backfill] {len(df)} games for {date_str}")
    return df


def main() -> None:
    dates = _parse_dates()

    repo_root = Path.cwd()
    sched_dir = repo_root / "data" / "schedules"
    sched_dir.mkdir(parents=True, exist_ok=True)

    out_path = sched_dir / "nba_schedule.csv"

    new_rows: List[pd.DataFrame] = []
    for d in dates:
        df_d = fetch_games_for_date(d)
        new_rows.append(df_d)

    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
    else:
        print("[backfill] No new rows to add.")
        return

    if out_path.exists():
        df_old = pd.read_csv(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates().reset_index(drop=True)
    else:
        df_all = df_new

    df_all.to_csv(out_path, index=False)
    print(f"[backfill] Wrote {len(df_all)} total rows to {out_path}")


if __name__ == "__main__":
    main()
