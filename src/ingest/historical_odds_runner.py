"""
Historical Odds Runner (CLOSE only)

Fetches historical NBA odds (spread + total + moneyline) from The Odds API
and normalizes them using normalize_odds_list().

Outputs:
    data/_snapshots/close_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests

from src.ingest.odds_normalizer import normalize_odds_list


# -----------------------
# Config
# -----------------------

SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "spreads,totals,h2h"
ODDS_FORMAT = "american"

BASE_URL = "https://api.the-odds-api.com/v4/sports"
API_KEY = os.getenv("ODDS_API_KEY")

RAW_DIR = Path("data/_snapshots/raw")
NORM_DIR = Path("data/_snapshots")

logger = logging.getLogger(__name__)


# -----------------------
# Helpers
# -----------------------

def parse_date(d: str):
    return datetime.strptime(d, "%Y-%m-%d").date()


def daterange(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def fetch_odds_for_date(run_date: str) -> List[Dict]:
    """
    Fetch closing odds snapshot for a specific date.
    """
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in environment")

    url = f"{BASE_URL}/{SPORT}/odds"

    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "date": run_date,
    }

    logger.info("Fetching odds for %s", run_date)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    return resp.json()


def write_raw_snapshot(data: List[Dict], run_date: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RAW_DIR / f"raw_{run_date}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)

    logger.info("Wrote raw odds snapshot %s", out_path)
    return out_path


def normalize_snapshot(odds: List[Dict], run_date: str) -> Path:
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


# -----------------------
# Main runner
# -----------------------

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
            odds = fetch_odds_for_date(run_date)
            write_raw_snapshot(odds, run_date)
            normalize_snapshot(odds, run_date)
        except Exception as e:
            logger.warning("Failed odds fetch for %s: %s", run_date, e)

    logger.info("Historical odds run complete.")


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch + normalize historical NBA odds (CLOSE)."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    run_range(
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
