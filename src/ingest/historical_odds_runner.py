"""
Historical Odds Runner (CLOSE only)

Fetches historical NBA odds (spread + total + moneyline) from The Odds API
(HISTORICAL endpoint) and normalizes them using normalize_odds_list().

Outputs:
    data/_snapshots/close_YYYYMMDD.csv
    data/_snapshots/raw/raw_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone, date as date_type
from pathlib import Path
from typing import List, Dict, Any

import requests

from src.ingest.odds_normalizer import normalize_odds_list


# -----------------------
# Config
# -----------------------

SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "spreads,totals,h2h"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

# IMPORTANT: historical endpoint (NOT /v4/sports/.../odds)
BASE_URL = "https://api.the-odds-api.com/v4/historical/sports"

API_KEY = os.getenv("ODDS_API_KEY")

RAW_DIR = Path("data/_snapshots/raw")
NORM_DIR = Path("data/_snapshots")

logger = logging.getLogger(__name__)


# -----------------------
# Helpers
# -----------------------

def parse_date(d: str) -> date_type:
    return datetime.strptime(d, "%Y-%m-%d").date()


def daterange(start: date_type, end: date_type):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _iso_z(dt: datetime) -> str:
    """ISO8601 with Z (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_historical_odds_for_date(run_date: str) -> List[Dict[str, Any]]:
    """
    Fetch odds for games on `run_date` using the HISTORICAL odds endpoint.

    Strategy:
      - Query a snapshot timestamp near end-of-day UTC (23:59:59Z) so we get
        the latest available odds snapshot at or before that time.
      - Constrain events to the day using commenceTimeFrom/commenceTimeTo
        (ISO8601).
    """
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in environment")

    # Snapshot timestamp (must be ISO8601); pick end of day UTC
    day = datetime.strptime(run_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    snap_ts = _iso_z(day.replace(hour=23, minute=59, second=59))

    # Filter events to just this calendar day (UTC)
    commence_from = _iso_z(day.replace(hour=0, minute=0, second=0))
    commence_to = _iso_z(day.replace(hour=23, minute=59, second=59))

    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
        # historical snapshot timestamp:
        "date": snap_ts,
        # constrain events to this day:
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }

    logger.info("Fetching HISTORICAL odds for %s (snapshot=%s)", run_date, snap_ts)
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()

    payload = resp.json()
    # Historical endpoints wrap results in {"timestamp":..., "data":[...]}
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Unexpected historical odds payload shape (missing list 'data').")
    return data


def write_raw_snapshot(data: List[Dict[str, Any]], run_date: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"raw_{run_date}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    logger.info("Wrote raw odds snapshot %s", out_path)
    return out_path


def normalize_snapshot(odds: List[Dict[str, Any]], run_date: str) -> Path:
    """
    Normalize odds list into CLOSE snapshot CSV using existing normalizer.
    """
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = NORM_DIR / f"close_{run_date.replace('-', '')}.csv"

    df = normalize_odds_list(
        odds,
        snapshot_type="close",
    )

    if df.empty:
        raise ValueError("Normalized odds DataFrame is empty")

    df.to_csv(out_csv, index=False)
    logger.info("Wrote normalized CLOSE odds %s (%d rows)", out_csv, len(df))
    return out_csv


def run_range(start: str, end: str, overwrite: bool):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [historical_odds] %(message)s",
    )

    start_d = parse_date(start)
    end_d = parse_date(end)

    logger.info("Running historical odds fetch %s → %s", start, end)

    for d in daterange(start_d, end_d):
        run_date = d.strftime("%Y-%m-%d")
        out_csv = NORM_DIR / f"close_{run_date.replace('-', '')}.csv"

        if out_csv.exists() and not overwrite:
            logger.info("Skipping %s (already exists)", run_date)
            continue

        try:
            odds = fetch_historical_odds_for_date(run_date)
            write_raw_snapshot(odds, run_date)
            normalize_snapshot(odds, run_date)
        except Exception as e:
            logger.warning("Failed odds fetch for %s: %s", run_date, e)

    logger.info("Historical odds run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch + normalize historical NBA odds (CLOSE)."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    run_range(start=args.start, end=args.end, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
