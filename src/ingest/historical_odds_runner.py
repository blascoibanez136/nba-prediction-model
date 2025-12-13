"""
Historical Odds Runner with multi‑snapshot fallback.

This script fetches historical NBA odds (spread, total, moneyline) from
The Odds API using the ``/v4/historical/sports`` endpoint and normalizes
them into wide "CLOSE" snapshots for downstream market analyses.

The runner no longer assumes that an end‑of‑day (23:59:59 UTC) snapshot
contains complete market data for older seasons. Instead, it attempts a
ladder of snapshot times (configured via ``SNAPSHOT_TIMES``) for each date
and selects the snapshot that returns the most games. Earlier times
(e.g. 20:00:00 or 16:00:00 UTC) often produce better coverage when
markets close early or data is missing at the nominal "close".

If the normalizer returns an empty DataFrame (common for early seasons or
when markets are absent), the runner falls back to a wide historical
builder (``build_wide_snapshot_from_raw``) that extracts whatever
information is available so downstream market aggregation can still
proceed. If both normalizers return empty DataFrames then no CSV is
written.

Outputs:
    data/_snapshots/close_YYYYMMDD.csv      Normalized or wide snapshot
    data/_snapshots/raw/raw_YYYY-MM-DD.json  Raw Odds API response (if data exists)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone, date as date_type
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# When fetching historical odds we may need to try multiple snapshot times on the same
# day. ``SNAPSHOT_TIMES`` defines a descending ladder of times (in UTC) to
# attempt. The runner will choose the snapshot with the most games in the
# response. These times should be in ``HH:MM:SS`` format and ordered from
# latest to earliest.
SNAPSHOT_TIMES: List[str] = [
    "23:59:59",  # typical end‑of‑day close
    "20:00:00",  # early evening (approx 3 hours before close)
    "16:00:00",  # late afternoon (approx 7 hours before close)
]

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


def fetch_historical_odds_for_date(run_date: str, *, snapshot_time: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch odds for games on ``run_date`` using The Odds API historical endpoint.

    A snapshot time (HH:MM:SS) may be provided to query the state of the market
    at a specific moment in UTC on the given date. When no snapshot time is
    supplied, the function defaults to end‑of‑day (23:59:59). The API is
    constrained to return only events commencing on the specified calendar day.

    Args:
        run_date: Date string in ``YYYY-MM-DD`` format.
        snapshot_time: Optional snapshot time (``HH:MM:SS``). If ``None`` the
            snapshot defaults to ``23:59:59``.

    Returns:
        A list of games (raw Odds API objects) for the requested date.
    """
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in environment")

    # Normalise snapshot_time; default to end of day
    if snapshot_time is None:
        snapshot_time = "23:59:59"
    try:
        hh, mm, ss = map(int, snapshot_time.split(":"))
    except Exception:
        raise ValueError(f"Invalid snapshot_time format: {snapshot_time}. Expected HH:MM:SS")

    # Build snapshot timestamp in UTC
    day = datetime.strptime(run_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    snap_ts = _iso_z(day.replace(hour=hh, minute=mm, second=ss))

    # Constrain events to this calendar day (UTC). We always bound from 00:00 to 23:59
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

    logger.info(
        "Fetching HISTORICAL odds for %s (snapshot=%s)", run_date, snap_ts
    )
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()

    payload = resp.json()
    # Historical endpoints wrap results in {"timestamp":..., "data":[...]}
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError(
            "Unexpected historical odds payload shape (missing list 'data')."
        )
    return data


def write_raw_snapshot(data: List[Dict[str, Any]], run_date: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"raw_{run_date}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    logger.info("Wrote raw odds snapshot %s", out_path)
    return out_path


def normalize_snapshot(odds: List[Dict[str, Any]], run_date: str) -> Optional[Path]:
    """
    Normalize odds list into CLOSE snapshot CSV using existing normalizer.

    If the normalizer returns an empty DataFrame (common for older seasons
    with missing markets), attempt to build a wide snapshot via
    ``build_wide_snapshot_from_raw``. If both normalizers return empty
    DataFrames no file is written and ``None`` is returned.
    """
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = NORM_DIR / f"close_{run_date.replace('-', '')}.csv"

    df = normalize_odds_list(
        odds,
        snapshot_type="close",
    )

    # If the normalizer returned no rows, fall back to a wide historical builder.
    if df.empty:
        logger.warning(
            "Normalized odds empty for %s. Raw games=%d. Likely missing bookmakers/markets for historical season.",
            run_date,
            len(odds),
        )
        try:
            from src.ingest.historical_wide_builder import build_wide_snapshot_from_raw
        except Exception as e:
            # Re‑raise import errors to surface configuration problems
            raise e
        df = build_wide_snapshot_from_raw(odds, snapshot_type="close")
        if df.empty:
            logger.warning(
                "Wide fallback also produced empty DataFrame for %s (games=%d).",
                run_date,
                len(odds),
            )
            return None

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
            # Attempt multiple snapshot times, selecting the snapshot with the
            # greatest number of games. Start with the latest time and fall
            # back earlier only if needed.
            best_odds: List[Dict[str, Any]] = []
            best_snap: Optional[str] = None
            for snap_time in SNAPSHOT_TIMES:
                try:
                    current = fetch_historical_odds_for_date(
                        run_date, snapshot_time=snap_time
                    )
                except Exception as e:
                    # Log and continue to next snapshot time on failure
                    logger.warning(
                        "Snapshot %s fetch for %s raised %s; trying earlier snapshot", snap_time, run_date, e
                    )
                    continue
                # Choose the snapshot with the most games. If equal length,
                # prefer the earlier (first) snapshot we attempted.
                if len(current) > len(best_odds):
                    best_odds = current
                    best_snap = snap_time
                # If we already have a non‑empty snapshot and the current
                # candidate returns no games, there is no point in continuing.
                # We continue through all times so we pick the maximum count.
            if best_snap is None:
                # None of the snapshot times succeeded; use empty list to run
                # fallback normalization and logging. No raw file will be
                # written if there is no data.
                odds = []
                selected_snap = SNAPSHOT_TIMES[0]
            else:
                odds = best_odds
                selected_snap = best_snap
            logger.info(
                "Selected snapshot %s for %s with %d games (from %s attempts)",
                selected_snap,
                run_date,
                len(odds),
                ", ".join(SNAPSHOT_TIMES),
            )
            # Write raw snapshot and normalize
            if odds:
                write_raw_snapshot(odds, run_date)
            csv_path = normalize_snapshot(odds, run_date)
            # Do not log a success if no CSV was produced
            if csv_path is None:
                logger.info("No data written for %s due to empty odds.", run_date)
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
