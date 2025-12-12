"""
Historical Prediction Runner for NBA Pro-Lite.

Purpose:
    Generate model predictions for historical NBA games using the
    same feature + model pipeline as run_daily.py.

Outputs:
    outputs/predictions_YYYY-MM-DD.csv

Phase 1:
    - Model-only predictions (no odds, no market ensemble)

Phase 2 (later):
    - Add historical odds
    - Generate predictions_YYYY-MM-DD_market.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from src.ingest.team_normalizer import normalize_team_name
from src.features.feature_builder import build_features
from src.model.model_predict import run_model_predictions


OUTPUT_DIR = "outputs"
HISTORY_DEFAULT = "data/history/games_2019_2024.csv"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def parse_date(d: str) -> datetime.date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def daterange(start: datetime.date, end: datetime.date) -> Iterable[datetime.date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------


def load_historical_schedule(path: str) -> pd.DataFrame:
    """
    Load historical games file.

    Required columns:
        game_date, home_team, away_team
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Historical games file not found: {path}")

    df = pd.read_csv(path)

    required = {"game_date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical games file missing columns: {missing}")

    df["game_date"] = df["game_date"].astype(str)
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    return df


def build_daily_schedule(history: pd.DataFrame, day: datetime.date) -> pd.DataFrame:
    """
    Build a schedule DataFrame for a single day that matches what
    feature_builder expects.
    """
    day_str = day.strftime("%Y-%m-%d")
    df = history[history["game_date"] == day_str].copy()

    if df.empty:
        return df

    # Match run_daily expectations
    df["game_date"] = day_str

    return df[["game_date", "home_team", "away_team"]]


def run_for_date(
    history: pd.DataFrame,
    day: datetime.date,
    overwrite: bool = False,
) -> None:
    day_str = day.strftime("%Y-%m-%d")
    out_path = os.path.join(OUTPUT_DIR, f"predictions_{day_str}.csv")

    if os.path.exists(out_path) and not overwrite:
        logger.info("Predictions already exist for %s — skipping.", day_str)
        return

    schedule = build_daily_schedule(history, day)
    if schedule.empty:
        logger.info("No games on %s — skipping.", day_str)
        return

    logger.info("Running historical predictions for %s (%d games)", day_str, len(schedule))

    # Build features using existing pipeline
    features = build_features(schedule)

    # Run trained models (win / spread / total)
    preds = run_model_predictions(features)

    # Safety check for backtest compatibility
    required_cols = {
        "game_date",
        "home_team",
        "away_team",
        "home_win_prob",
        "fair_spread",
    }
    missing = required_cols - set(preds.columns)
    if missing:
        raise ValueError(f"Prediction output missing columns: {missing}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preds.to_csv(out_path, index=False)

    logger.info("Wrote %s (%d rows)", out_path, len(preds))


def run_range(
    history_path: str,
    start: str,
    end: str,
    overwrite: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [historical] %(message)s",
    )

    history = load_historical_schedule(history_path)

    start_date = parse_date(start)
    end_date = parse_date(end)

    logger.info(
        "Generating historical predictions from %s → %s",
        start_date,
        end_date,
    )

    for day in daterange(start_date, end_date):
        run_for_date(history, day, overwrite=overwrite)

    logger.info("Historical prediction run complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical NBA predictions using Pro-Lite pipeline."
    )
    parser.add_argument(
        "--history",
        type=str,
        default=HISTORY_DEFAULT,
        help="Path to historical games CSV",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files",
    )

    args = parser.parse_args()

    run_range(
        history_path=args.history,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
