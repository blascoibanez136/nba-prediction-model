"""
Historical Odds Runner (multi-snapshot + NBA-date-safe window)

Fetches historical NBA odds (spread + total + moneyline) from The Odds API
historical endpoint, writes raw snapshots, and writes normalized CSV snapshots.

Key behaviors (LOCKED/PROFESSIONAL):
- Multi-snapshot fallback: tries candidate UTC snapshot times and chooses the
  snapshot that returns the most games for that date.
- Selection rule for "Earliest available with max coverage":
    1) maximize coverage (#games returned)
    2) tie-break by earliest snapshot time
- NBA-date-safe event window: many NBA games for a given "game_date" tip after
  midnight UTC (e.g., 7:30pm PT = 02:30Z next day). If we filter strictly to the
  UTC calendar day, we drop late games and coverage collapses.

  We therefore fetch events for a run_date in the window:
    commenceTimeFrom = run_date 00:00:00Z
    commenceTimeTo   = (run_date + 1 day) 11:59:59Z

- Normalization: uses normalize_odds_list(); if empty, falls back to a wide builder.
- Empty payloads never pass silently: we log explicitly and skip writing.

Token safety:
- Supports snapshot-types selection (open/close)
- Default behavior is skip-existing (won't refetch files that already exist)
- OPEN can be restricted to dates where CLOSE already exists (default True),
  preventing accidental token burn outside your known close range.
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


logger = logging.getLogger(__name__)

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

# CLOSE candidates (UTC). Keep your original list, highest-likelihood late coverage.
CLOSE_SNAPSHOT_TIMES = ["23:59:59", "20:00:00", "16:00:00"]

# OPEN candidates (UTC). "Early day" snapshots, then we pick earliest with max coverage.
# These are deliberately earlier than close and still likely to have markets.
OPEN_SNAPSHOT_TIMES = ["12:00:00", "14:00:00", "16:00:00", "18:00:00"]


# -----------------------
# Date helpers
# -----------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def _iso(d: date, hhmmss: str) -> str:
    return f"{d.isoformat()}T{hhmmss}Z"


def _nba_window_utc(run_date: date) -> Tuple[str, str]:
    # NBA-date-safe window: include next-day UTC morning to capture late tips.
    start = _iso(run_date, "00:00:00")
    end = _iso(run_date + timedelta(days=1), "11:59:59")
    return start, end


# -----------------------
# IO paths
# -----------------------

def _ymd(run_date: str) -> str:
    return run_date.replace("-", "")


def snapshot_csv_path(snapshot_type: str, run_date: str) -> Path:
    return SNAPSHOT_DIR / f"{snapshot_type}_{_ymd(run_date)}.csv"


def raw_json_path(snapshot_type: str, run_date: str) -> Path:
    return RAW_DIR / f"raw_{snapshot_type}_{run_date}.json"


def day_audit_path(snapshot_type: str, run_date: str) -> Path:
    return RAW_DIR / f"audit_{snapshot_type}_{run_date}.json"


# -----------------------
# Fetch
# -----------------------

@dataclass
class FetchResult:
    snapshot_time: str
    games: List[Dict[str, Any]]
    error: Optional[str] = None


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
        "date": _iso(d, snapshot_time),  # snapshot timestamp
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    games = payload.get("data", payload)  # some shapes return list directly
    if not isinstance(games, list):
        logger.warning("Unexpected payload shape for %s @ %s: %s", run_date, snapshot_time, type(games))
        return []
    return games


# -----------------------
# Write / Normalize
# -----------------------

def write_raw_snapshot(games: List[Dict[str, Any]], snapshot_type: str, run_date: str) -> Path:
    out = raw_json_path(snapshot_type, run_date)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(games, f)
    logger.info("Wrote raw odds snapshot %s", out)
    return out


def normalize_snapshot(games: List[Dict[str, Any]], snapshot_type: str, run_date: str) -> Optional[Path]:
    out_csv = snapshot_csv_path(snapshot_type, run_date)

    df = normalize_odds_list(games, snapshot_type=snapshot_type)

    if df is None or df.empty:
        logger.warning(
            "Normalized odds empty for %s (%s). Raw games=%d. Attempting wide fallback.",
            run_date,
            snapshot_type,
            len(games),
        )
        df = build_wide_snapshot_from_raw(games, snapshot_type=snapshot_type)

    if df is None or df.empty:
        logger.warning("Wide fallback also produced empty DataFrame for %s (%s) (games=%d).", run_date, snapshot_type, len(games))
        return None

    df.to_csv(out_csv, index=False)
    logger.info("Wrote normalized %s odds %s (%d rows)", snapshot_type.upper(), out_csv, len(df))
    return out_csv


# -----------------------
# Selection rule: earliest with max coverage
# -----------------------

def _candidate_times(snapshot_type: str) -> List[str]:
    if snapshot_type == "open":
        return OPEN_SNAPSHOT_TIMES
    if snapshot_type == "close":
        return CLOSE_SNAPSHOT_TIMES
    raise ValueError(f"Unsupported snapshot_type: {snapshot_type}")


def choose_best_attempt(attempts: List[FetchResult]) -> Optional[FetchResult]:
    """
    Choose by:
      1) max coverage (len(games))
      2) earliest snapshot_time as tie-break
    """
    good = [a for a in attempts if a.error is None]
    if not good:
        return None
    # Sort by (-coverage, snapshot_time asc)
    good_sorted = sorted(good, key=lambda a: (-len(a.games), a.snapshot_time))
    return good_sorted[0]


# -----------------------
# Runner
# -----------------------

def run_range(
    start: str,
    end: str,
    *,
    snapshot_types: List[str],
    overwrite: bool,
    skip_existing: bool,
    require_close_for_open: bool,
    max_requests: Optional[int],
) -> Dict[str, Any]:
    d0 = _parse_date(start)
    d1 = _parse_date(end)

    if d0 > d1:
        logger.warning("Start date %s is after end date %s; nothing to do.", start, end)
        return {"status": "noop", "reason": "start_after_end"}

    logger.info("Running historical odds fetch %s → %s (types=%s)", start, end, snapshot_types)

    run_audit: Dict[str, Any] = {
        "start": start,
        "end": end,
        "snapshot_types": snapshot_types,
        "overwrite": overwrite,
        "skip_existing": skip_existing,
        "require_close_for_open": require_close_for_open,
        "max_requests": max_requests,
        "days_considered": 0,
        "requests_made": 0,
        "written": [],
        "skipped": [],
        "errors": [],
    }

    req_count = 0

    for d in _iter_dates(d0, d1):
        run_date = d.isoformat()
        run_audit["days_considered"] += 1

        # For each snapshot type requested
        for stype in snapshot_types:
            out_csv = snapshot_csv_path(stype, run_date)
            close_csv = snapshot_csv_path("close", run_date)

            # Optional guardrail: only fetch OPEN where CLOSE already exists
            if stype == "open" and require_close_for_open and not close_csv.exists():
                run_audit["skipped"].append({
                    "run_date": run_date,
                    "snapshot_type": stype,
                    "reason": "require_close_for_open_true_and_close_missing",
                    "close_expected": str(close_csv),
                })
                logger.info("Skipping %s (%s) because close snapshot missing and require_close_for_open=True", run_date, stype)
                continue

            # Skip existing unless overwrite
            if skip_existing and out_csv.exists() and not overwrite:
                run_audit["skipped"].append({
                    "run_date": run_date,
                    "snapshot_type": stype,
                    "reason": "exists",
                    "path": str(out_csv),
                })
                logger.info("Skipping %s (%s) (snapshot exists)", run_date, stype)
                continue

            times = _candidate_times(stype)
            attempts: List[FetchResult] = []

            # Try candidate times
            for hhmmss in times:
                if max_requests is not None and req_count >= max_requests:
                    logger.warning("Reached max_requests_per_run=%s; stopping early.", max_requests)
                    run_audit["status"] = "stopped_max_requests"
                    run_audit["requests_made"] = req_count
                    (SNAPSHOT_DIR / "historical_odds_runner_audit.json").write_text(
                        json.dumps(run_audit, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    return run_audit

                try:
                    logger.info("Fetching HISTORICAL odds for %s (%s snapshot=%s)", run_date, stype, _iso(d, hhmmss))
                    games = fetch_historical_odds_for_date(run_date, hhmmss)
                    req_count += 1
                    attempts.append(FetchResult(snapshot_time=hhmmss, games=games))
                    logger.info("  -> %s (%s) got %d games", run_date, stype, len(games))
                except Exception as e:
                    req_count += 1
                    attempts.append(FetchResult(snapshot_time=hhmmss, games=[], error=repr(e)))
                    logger.warning("  -> %s (%s) error @ %s: %s", run_date, stype, hhmmss, e)

            best = choose_best_attempt(attempts)

            # Write day audit regardless
            day_audit = {
                "run_date": run_date,
                "snapshot_type": stype,
                "candidate_times": times,
                "attempts": [
                    {"snapshot_time": a.snapshot_time, "games": len(a.games), "error": a.error}
                    for a in attempts
                ],
                "selected": None,
                "written_csv": None,
                "written_raw": None,
            }

            if best is None or len(best.games) == 0:
                day_audit["selected"] = None
                run_audit["errors"].append({
                    "run_date": run_date,
                    "snapshot_type": stype,
                    "reason": "no_usable_payload",
                    "attempts": day_audit["attempts"],
                })
                day_audit_path(stype, run_date).write_text(json.dumps(day_audit, indent=2, sort_keys=True), encoding="utf-8")
                logger.warning("No usable payload for %s (%s). Skipping write.", run_date, stype)
                continue

            # Selected snapshot details
            day_audit["selected"] = {
                "snapshot_time": best.snapshot_time,
                "games": len(best.games),
                "rule": "earliest_with_max_coverage",
            }

            # Write raw + normalized
            rawp = write_raw_snapshot(best.games, stype, run_date)
            csvp = normalize_snapshot(best.games, stype, run_date)

            day_audit["written_raw"] = str(rawp)
            day_audit["written_csv"] = str(csvp) if csvp else None
            day_audit_path(stype, run_date).write_text(json.dumps(day_audit, indent=2, sort_keys=True), encoding="utf-8")

            if csvp is None:
                run_audit["errors"].append({
                    "run_date": run_date,
                    "snapshot_type": stype,
                    "reason": "normalization_empty",
                    "selected": day_audit["selected"],
                })
                continue

            run_audit["written"].append({
                "run_date": run_date,
                "snapshot_type": stype,
                "snapshot_time": best.snapshot_time,
                "games": len(best.games),
                "csv": str(csvp),
                "raw": str(rawp),
            })

    run_audit["requests_made"] = req_count
    (SNAPSHOT_DIR / "historical_odds_runner_audit.json").write_text(
        json.dumps(run_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Wrote run audit %s", SNAPSHOT_DIR / "historical_odds_runner_audit.json")
    return run_audit


# -----------------------
# CLI
# -----------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")

    # Behavior controls
    p.add_argument("--snapshot-types", default="close", help="Comma-separated: open,close. Example: open or open,close")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing snapshot CSVs (dangerous; off by default)")
    p.add_argument("--skip-existing", action="store_true", default=True, help="Skip if snapshot CSV exists (default True)")
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Disable skip-existing (will refetch unless prevented by overwrite logic).",
    )

    # Token-safety: only fetch OPEN on dates where CLOSE exists
    p.add_argument(
        "--require-close-for-open",
        action="store_true",
        default=True,
        help="Only fetch OPEN when close_YYYYMMDD.csv exists (default True).",
    )
    p.add_argument(
        "--no-require-close-for-open",
        action="store_true",
        help="Allow fetching OPEN even if close is missing.",
    )

    # Hard cap on requests for safety
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Hard cap on API requests made in this run (safety valve).",
    )
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args()

    snapshot_types = [s.strip().lower() for s in str(args.snapshot_types).split(",") if s.strip()]
    for st in snapshot_types:
        if st not in {"open", "close"}:
            raise SystemExit(f"Unsupported snapshot type: {st} (allowed: open, close)")

    skip_existing = bool(args.skip_existing) and not bool(args.no_skip_existing)
    require_close_for_open = bool(args.require_close_for_open) and not bool(args.no_require_close_for_open)

    run_range(
        args.start,
        args.end,
        snapshot_types=snapshot_types,
        overwrite=bool(args.overwrite),
        skip_existing=skip_existing,
        require_close_for_open=require_close_for_open,
        max_requests=args.max_requests,
    )


if __name__ == "__main__":
    main()
