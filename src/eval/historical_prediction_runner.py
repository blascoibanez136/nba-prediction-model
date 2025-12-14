from __future__ import annotations

import argparse
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import _merge_key

# Directories for input and output
SNAPSHOT_DIR = Path("data/_snapshots")
OUTPUT_DIR = Path("outputs")

logger = logging.getLogger("historical")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [historical] %(message)s",
)

# Coverage thresholds (match your contract)
COVERAGE_DISABLE_THRESHOLD = 0.80
COVERAGE_WARN_THRESHOLD = 0.95


def daterange(start: str, end: str):
    """Yield YYYY-MM-DD strings for each day in the inclusive range [start, end]."""
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    while d0 <= d1:
        yield d0.strftime("%Y-%m-%d")
        d0 += timedelta(days=1)


def load_games_for_date(history_df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Extract games for a specific run_date from the history DataFrame."""
    day = history_df[history_df["game_date"].astype(str) == run_date].copy()
    if day.empty:
        return pd.DataFrame(columns=["game_date", "home_team", "away_team"])

    # Only what predict_games needs
    out = day[["game_date", "home_team", "away_team"]].copy()
    out["game_date"] = run_date
    return out


def _find_latest_snapshot_for_date(run_date: str, snapshot_type: str = "close") -> Optional[Path]:
    """
    Locate the latest snapshot CSV for a given date.

    Accepts BOTH formats:
      - close_YYYYMMDD.csv
      - close_YYYYMMDD_HHMMSS.csv  (preferred / most common in this repo)

    Returns:
      Path to the newest matching file, or None.
    """
    ymd = run_date.replace("-", "")
    # Prefer timestamped (can be multiple snapshots per day)
    pattern_ts = str(SNAPSHOT_DIR / f"{snapshot_type}_{ymd}_*.csv")
    candidates = sorted(glob.glob(pattern_ts))

    # Fallback: non-timestamp single file
    if not candidates:
        pattern_plain = SNAPSHOT_DIR / f"{snapshot_type}_{ymd}.csv"
        if pattern_plain.exists():
            return pattern_plain
        return None

    # newest (lexicographic works for YYYYMMDD_HHMMSS)
    return Path(candidates[-1])


def run_day(history_df: pd.DataFrame, run_date: str, apply_market: bool, overwrite: bool) -> None:
    """
    Run predictions for a single day.

    - Writes base predictions: outputs/predictions_{run_date}.csv
    - Optionally writes market preds: outputs/predictions_{run_date}_market.csv
      (only if odds snapshot exists AND coverage passes thresholds)
    """
    games = load_games_for_date(history_df, run_date)
    if games.empty:
        logger.info("No games for %s — skipping", run_date)
        return

    logger.info("Running predictions for %s (%d games)", run_date, len(games))
    preds = predict_games(games)

    # Ensure required columns exist and compute merge_key
    preds["game_date"] = run_date
    preds["merge_key"] = preds.apply(
        lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]),
        axis=1,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    base_path = OUTPUT_DIR / f"predictions_{run_date}.csv"

    if base_path.exists() and not overwrite:
        logger.info("Base predictions already exist for %s — skipping write", run_date)
    else:
        preds.to_csv(base_path, index=False)
        logger.info("Wrote %s (%d rows)", base_path, len(preds))

    if not apply_market:
        return

    snap_path = _find_latest_snapshot_for_date(run_date, snapshot_type="close")
    if snap_path is None:
        logger.warning("No CLOSE odds snapshot found for %s in %s — skipping market ensemble", run_date, SNAPSHOT_DIR)
        return

    odds = pd.read_csv(snap_path)
    if odds.empty or "merge_key" not in odds.columns:
        logger.warning(
            "CLOSE snapshot invalid for %s (%s): empty=%s, has_merge_key=%s — skipping market ensemble",
            run_date,
            snap_path,
            odds.empty,
            "merge_key" in odds.columns,
        )
        return

    # Coverage: unique merge_keys present in odds vs schedule
    schedule_keys = set(preds["merge_key"].astype(str).tolist())
    odds_keys = set(odds["merge_key"].dropna().astype(str).tolist())

    covered = len(schedule_keys & odds_keys)
    total = len(schedule_keys)
    coverage = covered / total if total else 0.0

    logger.info(
        "Using CLOSE snapshot: %s | Odds coverage for %s: %d/%d games (%.1f%%)",
        snap_path,
        run_date,
        covered,
        total,
        coverage * 100.0,
    )

    if coverage < COVERAGE_DISABLE_THRESHOLD:
        logger.warning(
            "Coverage %.1f%% < %.0f%% for %s — disabling market ensemble for this day",
            coverage * 100.0,
            COVERAGE_DISABLE_THRESHOLD * 100.0,
            run_date,
        )
        return
    elif coverage < COVERAGE_WARN_THRESHOLD:
        logger.warning(
            "Coverage %.1f%% < %.0f%% for %s — proceeding, but results may be incomplete",
            coverage * 100.0,
            COVERAGE_WARN_THRESHOLD * 100.0,
            run_date,
        )

    market_preds = apply_market_ensemble(preds, odds)

    out_path = OUTPUT_DIR / f"predictions_{run_date}_market.csv"
    if out_path.exists() and not overwrite:
        logger.info("Market predictions already exist for %s — skipping write", run_date)
        return

    market_preds.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(market_preds))


def main() -> None:
    """
    Example:

      python -m src.eval.historical_prediction_runner \
        --history data/history/games_2019_2024.csv \
        --start 2023-10-24 \
        --end 2024-06-17 \
        --apply-market \
        --overwrite

    History CSV must contain:
      - game_date
      - home_team
      - away_team
    (It may contain scores too; we ignore them here.)
    """
    ap = argparse.ArgumentParser(
        description="Run historical model predictions (optionally market-adjusted)."
    )
    ap.add_argument("--history", required=True, help="Path to games history CSV (with game_date, home_team, away_team)")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--apply-market", action="store_true", help="Apply market ensemble using CLOSE snapshots if available")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    history_df = pd.read_csv(args.history)
    required_cols = {"game_date", "home_team", "away_team"}
    missing = required_cols - set(history_df.columns)
    if missing:
        raise ValueError(f"History CSV missing required columns: {missing}")

    logger.info(
        "Historical predictions %s → %s (apply_market=%s overwrite=%s)",
        args.start,
        args.end,
        args.apply_market,
        args.overwrite,
    )

    for d in daterange(args.start, args.end):
        run_day(history_df, d, args.apply_market, args.overwrite)

    logger.info("Historical prediction run complete.")


if __name__ == "__main__":
    main()
