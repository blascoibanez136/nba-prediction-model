"""
Historical Prediction Runner for NBA Pro-Lite.

Generates historical model predictions using the SAME
predict_games() pipeline as run_daily.py.

Outputs:
    outputs/predictions_YYYY-MM-DD.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from src.model.predict import predict_games
from src.eval.edge_picker import _merge_key as ep_merge_key
from src.ingest.team_normalizer import normalize_team_name


OUTPUT_DIR = "outputs"
HISTORY_DEFAULT = "data/history/games_2019_2024.csv"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_date(d: str):
    return datetime.strptime(d, "%Y-%m-%d").date()


def daterange(start, end) -> Iterable:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def load_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")

    df = pd.read_csv(path)

    required = {"game_date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"History file missing columns: {missing}")

    df["game_date"] = df["game_date"].astype(str)
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    return df


def build_daily_games(history: pd.DataFrame, day) -> pd.DataFrame:
    day_str = day.strftime("%Y-%m-%d")
    games = history[history["game_date"] == day_str].copy()

    if games.empty:
        return games

    games["game_date"] = day_str

    # Match predict_games() expectations
    return games[["game_date", "home_team", "away_team"]]


def run_for_day(history: pd.DataFrame, day, overwrite: bool):
    day_str = day.strftime("%Y-%m-%d")
    out_path = f"{OUTPUT_DIR}/predictions_{day_str}.csv"

    if os.path.exists(out_path) and not overwrite:
        logger.info("Skipping %s (already exists)", day_str)
        return

    games_df = build_daily_games(history, day)
    if games_df.empty:
        logger.info("No games on %s", day_str)
        return

    logger.info("Running predictions for %s (%d games)", day_str, len(games_df))

    preds = predict_games(games_df)

    if "game_date" not in preds.columns:
        preds["game_date"] = day_str

    preds["merge_key"] = preds.apply(
        lambda r: ep_merge_key(r["home_team"], r["away_team"], r["game_date"]),
        axis=1,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preds.to_csv(out_path, index=False)

    logger.info("Wrote %s (%d rows)", out_path, len(preds))


def run_range(history_path: str, start: str, end: str, overwrite: bool):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [historical] %(message)s",
    )

    history = load_history(history_path)

    start_d = parse_date(start)
    end_d = parse_date(end)

    logger.info("Historical predictions %s → %s", start, end)

    for d in daterange(start_d, end_d):
        run_for_day(history, d, overwrite)

    logger.info("Historical prediction run complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate historical NBA predictions using Pro-Lite pipeline."
    )
    parser.add_argument("--history", default=HISTORY_DEFAULT)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    run_range(
        history_path=args.history,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
