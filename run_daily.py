"""
run_daily.py

End-to-end daily NBA Pro-Lite pipeline.

Schedule logic (in order of preference):
1) Try to fetch today's NBA games from balldontlie.io.
2) If none are found, fall back to a pre-loaded official schedule CSV:
   data/schedules/nba_schedule.csv
3) If still no games, fall back to inferring the slate from
   outputs/odds_dispersion_latest.csv.

Then:
- Use the trained models to generate predictions (predict_games).
- Add merge_key for joining with odds.
- Apply market_ensemble using the latest normalized CLOSE odds snapshot.
- Run edge_picker to produce:
    - outputs/picks_<date>.csv
    - picks_report.html
- Write run_summary.md.

Environment:
- RUN_DATE            : optional override (YYYY-MM-DD); default is today's date (UTC).
- BALLDONTLIE_API_KEY : optional API key for balldontlie.io (Authorization: Bearer ...).
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import main as run_edge_picker, _merge_key as _ep_merge_key


# -----------------------
# logging setup
# -----------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    # Basic configuration for script-style usage.
    # GitHub Actions will capture stdout.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [run_daily] %(message)s",
    )

BALD_BASE_URL = "https://api.balldontlie.io/v1"
BALD_API_KEY = os.getenv("BALLDONTLIE_API_KEY")

SNAPSHOT_DIR = Path("data") / "_snapshots"


def _get_run_date() -> str:
    """Resolve the run date from env or use today's date."""
    return os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")


# -----------------------
# Schedule fetchers
# -----------------------

def fetch_today_games_from_balldontlie(run_date: str) -> pd.DataFrame:
    """
    Fetch today's NBA games from balldontlie.io /games endpoint.

    Endpoint docs (v1):
      GET /games?dates[]=YYYY-MM-DD&per_page=100

    Returns DataFrame with:
      game_id, game_date, home_team, away_team

    (May be empty if the API has no games for that date.)
    """
    params = {
        "dates[]": run_date,
        "per_page": 100,
    }
    headers: Dict[str, str] = {}
    if BALD_API_KEY:
        headers["Authorization"] = f"Bearer {BALD_API_KEY}"

    url = f"{BALD_BASE_URL}/games"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("balldontlie schedule fetch failed for %s: %s", run_date, e)
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    data = resp.json().get("data", [])
    rows: List[Dict[str, Any]] = []

    for g in data:
        gid = g.get("id")
        home = g.get("home_team", {}) or {}
        away = g.get("visitor_team", {}) or {}
        home_name = home.get("full_name") or home.get("name")
        away_name = away.get("full_name") or away.get("name")
        if not home_name or not away_name:
            continue

        d = g.get("date") or ""  # Typically ISO like "2025-12-07T00:00:00.000Z"
        game_date = d[:10] if d else run_date

        rows.append(
            {
                "game_id": gid,
                "game_date": game_date,
                "home_team": home_name,
                "away_team": away_name,
            }
        )

    df = pd.DataFrame(rows, columns=["game_id", "game_date", "home_team", "away_team"])
    if df.empty:
        logger.info("balldontlie returned no games for %s.", run_date)
    else:
        logger.info("Fetched %d games from balldontlie for %s.", len(df), run_date)
    return df


def fetch_today_games_from_schedule_csv(
    run_date: str,
    csv_path: str = "data/schedules/nba_schedule.csv",
) -> pd.DataFrame:
    """
    Fallback: use a pre-loaded official schedule CSV.

    CSV is expected to have at least:
      game_date, home_team, away_team

    Optional:
      game_id

    Returns DataFrame with:
      game_id, game_date, home_team, away_team
    """
    if not os.path.exists(csv_path):
        logger.info("No schedule CSV found at %s.", csv_path)
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    sched = pd.read_csv(csv_path)
    required = {"game_date", "home_team", "away_team"}
    missing = required - set(sched.columns)
    if missing:
        logger.warning("schedule CSV missing columns: %s", missing)
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    day = sched[sched["game_date"] == run_date].copy()
    if day.empty:
        logger.info("schedule CSV has no rows for %s.", run_date)
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    if "game_id" not in day.columns:
        # Synthetic ID if not provided
        day["game_id"] = day.apply(
