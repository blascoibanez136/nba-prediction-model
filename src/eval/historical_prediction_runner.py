from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import _merge_key
from src.ingest.historical_schedule_loader import load_games_for_date

SNAPSHOT_DIR = Path("data/_snapshots")
OUTPUT_DIR = Path("outputs")

logger = logging.getLogger("historical")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [historical] %(message)s",
)


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    while d0 <= d1:
        yield d0.strftime("%Y-%m-%d")
        d0 += timedelta(days=1)


def run_day(run_date: str, apply_market: bool, overwrite: bool):
    games = load_games_for_date(run_date)
    if games.empty:
        logger.info("No games for %s — skipping", run_date)
        return

    logger.info("Running predictions for %s (%d games)", run_date, len(games))

    preds = predict_games(games)
    preds["game_date"] = run_date
    preds["merge_key"] = preds.apply(
        lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]),
        axis=1,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    base_path = OUTPUT_DIR / f"predictions_{run_date}.csv"

    if base_path.exists() and not overwrite:
        logger.info("Base predictions already exist — skipping")
        return

    preds.to_csv(base_path, index=False)
    logger.info("Wrote %s (%d rows)", base_path, len(preds))

    if not apply_market:
        return

    close_csv = SNAPSHOT_DIR / f"close_{run_date.replace('-', '')}.csv"
    if not close_csv.exists():
        logger.warning("No CLOSE odds for %s — skipping market ensemble", run_date)
        return

    odds = pd.read_csv(close_csv)
    market_preds = apply_market_ensemble(preds, odds)

    out_path = OUTPUT_DIR / f"predictions_{run_date}_market.csv"
    market_preds.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(market_preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logger.info(
        "Historical predictions %s → %s (market=%s)",
        args.start,
        args.end,
        args.apply_market,
    )

    for d in daterange(args.start, args.end):
        run_day(d, args.apply_market, args.overwrite)

    logger.info("Historical prediction run complete.")


if __name__ == "__main__":
    main()
