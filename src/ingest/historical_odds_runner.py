"""
Historical Odds Runner (multi-snapshot + NBA-date-safe window)

Fetches historical NBA odds (spread + total + moneyline) from The Odds API
historical endpoint, writes raw snapshots, and writes normalized CLOSE CSVs.

Key behaviors:
- Multi-snapshot fallback: tries SNAPSHOT_TIMES (UTC) and chooses the snapshot
  that returns the most games for that date.
- NBA-date-safe event window: many NBA games for a given "game_date" tip after
  midnight UTC (e.g., 7:30pm PT = 02:30Z next day). If we filter strictly to the
  UTC calendar day, we drop late games and coverage collapses.

  We therefore fetch events for a run_date in the window:
    commenceTimeFrom = run_date 00:00:00Z
    commenceTimeTo   = (run_date + 1 day) 11:59:59Z

  This captures all games belonging to the NBA local date without leaking into
  the next day's slate.

- Normalization: uses normalize_odds_list(); if empty, falls back to a wide
  builder that extracts what it can from raw payloads.
- Empty payloads never pass silently: we log explicitly and skip writing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.ingest.odds_normalizer import normalize_odds_list
from src.ingest.historical_wide_builder import build_wide_snapshot_from_raw

logger = logging.getLogger("historical_odds")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

# -----------------------
# Config
# -----------------------

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports"
SPORT_KEY = os.getenv("ODDS_SPORT_KEY", "basketball_nba")

REGIONS = os.getenv("ODDS_REGIONS", "us")
MARKETS = os.getenv("ODDS_MARKETS", "h2h,spreads,totals")
ODDS_FORMAT = os.getenv("ODDS_FORMAT", "american")
DATE_FORMAT = "iso"

SNAPSHOT_DIR = Path("data/_snapshots")
RAW_DIR = SNAPSHOT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Try multiple snapshot times (UTC) and choose the response with the most games.
SNAPSHOT_TIMES = ["23:59:59", "20:00:00", "16:00:00"]


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _iso(ts_date: date, hhmmss: str) -> str:
    return f"{ts_date.isoformat()}T{hhmmss}Z"


def _nba_window_utc(run_date: date) -> Tuple[str, str]:
    """
    NBA-date-safe event window in UTC.

    Many NBA games on `run_date` (NBA local date) start after midnight UTC.
    So we fetch up through next-day 11:59:59Z (7:59am ET).
    """
    start = _iso(run_date, "00:00:00")
    end = _iso(run_date + timedelta(days=1), "11:59:59")
    return start, end


@dataclass
class FetchResult:
    snapshot_time: str
    games: List[Dict[str, Any]]


def fetch_historical_odds_for_date(run_date: str, snapshot_time: str) -> List[Dict[str, Any]]:
    """
    Fetch historical odds for games belonging to NBA game_date `run_date`,
    using snapshot timestamp `run_date` + snapshot_time (UTC).

    Returns a list of game dicts (payload['data'] if present).
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY env var not set")

    d = _parse_date(run_date)
    commence_from, commence_to = _nba_window_utc(d)

    url = f"{BASE_URL}/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
        # snapshot timestamp
        "date": _iso(d, snapshot_time),
        # event window
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # Historical endpoint payload: {"timestamp": "...", "data": [ ... ]}
    games = payload.get("data", [])
    if not isinstance(games, list):
        logger.warning("Unexpected payload shape for %s @ %s: %s", run_date, snapshot_time, type(games))
        return []
    return games


def write_raw_snapshot(games: List[Dict[str, Any]], run_date: str) -> Path:
    out = RAW_DIR / f"raw_{run_date}.json"
    with open(out, "w") as f:
        json.dump(games, f)
    logger.info("Wrote raw odds snapshot %s", out)
    return out


def normalize_snapshot(games: List[Dict[str, Any]], run_date: str) -> Optional[Path]:
    out_csv = SNAPSHOT_DIR / f"close_{run_date.replace('-', '')}.csv"

    df = normalize_odds_list(games, snapshot_type="close")

    if df is None or df.empty:
        logger.warning(
            "Normalized odds empty for %s. Raw games=%d. Attempting wide fallback.",
            run_date,
            len(games),
        )
        df = build_wide_snapshot_from_raw(games, snapshot_type="close")

    if df is None or df.empty:
        logger.warning("Wide fallback also produced empty DataFrame for %s (games=%d).", run_date, len(games))
        return None

    df.to_csv(out_csv, index=False)
    logger.info("Wrote normalized CLOSE odds %s (%d rows)", out_csv, len(df))
    return out_csv


def run_range(start: str, end: str, overwrite: bool):
    d0 = _parse_date(start)
    d1 = _parse_date(end)

    if d0 > d1:
        logger.warning("Start date %s is after end date %s; nothing to do.", start, end)
        return

    logger.info("Running historical odds fetch %s \u2192 %s", start, end)

    for d in _iter_dates(d0, d1):
        run_date = d.isoformat()
        out_csv = SNAPSHOT_DIR / f"close_{run_date.replace('-', '')}.csv"
        raw_path = RAW_DIR / f"raw_{run_date}.json"

        if out_csv.exists() and not overwrite:
            logger.info("Skipping %s (close snapshot exists)", run_date)
            continue

        attempts: List[FetchResult] = []
        for hhmmss in SNAPSHOT_TIMES:
            try:
                logger.info("Fetching HISTORICAL odds for %s (snapshot=%s)", run_date, _iso(d, hhmmss))
                games = fetch_historical_odds_for_date(run_date, hhmmss)
                attempts.append(FetchResult(snapshot_time=hhmmss, games=games))
            except Exception as e:
                logger.warning("Failed odds fetch for %s @ %s: %s", run_date, hhmmss, e)

        if not attempts:
            logger.warning("No snapshot attempts succeeded for %s", run_date)
            continue

        # Choose snapshot that returns the most games
        attempts_sorted = sorted(attempts, key=lambda r: len(r.games), reverse=True)
        best = attempts_sorted[0]
        logger.info(
            "Selected snapshot %s for %s with %d games (from %s attempts)",
            best.snapshot_time,
            run_date,
            len(best.games),
            ", ".join([a.snapshot_time for a in attempts]),
        )

        # Write raw (best) and normalize
        write_raw_snapshot(best.games, run_date)
        csv_path = normalize_snapshot(best.games, run_date)
        if csv_path is None:
            logger.info("No data written for %s due to empty odds.", run_date)

    logger.info("Historical odds run complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing close CSVs")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_range(args.start, args.end, args.overwrite)


if __name__ == "__main__":
    main()
