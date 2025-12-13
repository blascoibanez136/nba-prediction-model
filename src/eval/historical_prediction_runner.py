from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

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


def daterange(start: str, end: str):
    """Yield YYYY-MM-DD strings for each day in the inclusive range [start, end]."""
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    while d0 <= d1:
        yield d0.strftime("%Y-%m-%d")
        d0 += timedelta(days=1)


def load_games_for_date(history_df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Extract games for a specific run_date from the history DataFrame."""
    day = history_df[history_df["game_date"] == run_date].copy()
    if day.empty:
        return pd.DataFrame(columns=["game_date", "home_team", "away_team"])
    # Only what predict_games needs
    return day[["game_date", "home_team", "away_team"]].copy()


def run_day(history_df: pd.DataFrame, run_date: str, apply_market: bool, overwrite: bool):
    """
    Run predictions for a single day.

    Parameters
    ----------
    history_df : pd.DataFrame
        DataFrame containing historical game schedule with columns game_date, home_team, away_team.
    run_date : str
        Date to run predictions for (YYYY-MM-DD).
    apply_market : bool
        Whether to attempt market ensemble adjustment using odds snapshots.
    overwrite : bool
        If True, overwrite existing prediction outputs.

    This function will compute coverage of odds snapshots and decide whether to apply
    market ensemble based on configurable thresholds. Coverage <80% disables
    ensemble, 80%–95% warns but proceeds, >=95% proceeds normally.
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
        logger.info("Base predictions already exist for %s — skipping", run_date)
    else:
        preds.to_csv(base_path, index=False)
        logger.info("Wrote %s (%d rows)", base_path, len(preds))

    # If the caller does not want market ensemble, stop here
    if not apply_market:
        return

    close_csv = SNAPSHOT_DIR / f"close_{run_date.replace('-', '')}.csv"
    if not close_csv.exists():
        logger.warning(
            "No CLOSE odds CSV for %s (%s) — skipping market ensemble", run_date, close_csv
        )
        return

    # Load odds snapshot (per-book wide format)
    odds = pd.read_csv(close_csv)

    # Compute coverage: unique merge_keys present in odds vs schedule
    schedule_keys = set(preds["merge_key"].tolist())
    odds_keys = set(odds["merge_key"].dropna().astype(str).unique().tolist())
    covered = len(schedule_keys & odds_keys)
    total = len(schedule_keys)
    coverage = covered / total if total else 0.0
    logger.info(
        "Odds coverage for %s: %d/%d games (%.1f%%)",
        run_date,
        covered,
        total,
        coverage * 100,
    )

    # Enforce merge coverage thresholds
    if coverage < 0.80:
        logger.warning(
            "Coverage %.1f%% < 80%% for %s — disabling market ensemble", coverage * 100, run_date
        )
        return
    elif coverage < 0.95:
        logger.warning(
            "Coverage %.1f%% < 95%% for %s — proceeding with market ensemble, results may be incomplete",
            coverage * 100,
            run_date,
        )

    # Apply market ensemble
    market_preds = apply_market_ensemble(preds, odds)

    out_path = OUTPUT_DIR / f"predictions_{run_date}_market.csv"
    market_preds.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(market_preds))


def main():
    """
    CLI entry point. Example usage:

        python -m src.eval.historical_prediction_runner \
            --history data/games_2019_2024.csv \
            --start 2021-10-19 \
            --end 2023-06-15 \
            --apply-market \
            --overwrite

    The history CSV must contain at least columns: game_date, home_team, away_team.
    """
    ap = argparse.ArgumentParser(
        description="Run historical model predictions (optionally market-adjusted)."
    )
    ap.add_argument("--history", required=True, help="Path to games history CSV (with game_date, home_team, away_team)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    history_df = pd.read_csv(args.history)
    required_cols = {"game_date", "home_team", "away_team"}
    missing = required_cols - set(history_df.columns)
    if missing:
        raise ValueError(f"History CSV missing required columns: {missing}")

    logger.info(
        "Historical predictions %s → %s (apply_market=%s)",
        args.start,
        args.end,
        args.apply_market,
    )

    for d in daterange(args.start, args.end):
        run_day(history_df, d, args.apply_market, args.overwrite)

    logger.info("Historical prediction run complete.")


if __name__ == "__main__":
    main()
