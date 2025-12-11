"""
Edge picker for NBA Pro-Lite / Elite.

Reads the latest market-adjusted predictions CSV, joins it with the latest
CLOSE odds snapshot, evaluates edges on spreads, and emits a pick sheet:

    outputs/picks_YYYY-MM-DD.csv

with one recommended spread pick per game (or zero if no edges qualify).

Columns in the pick sheet:

    - game_id
    - game_date
    - book
    - market_side             ("home" or "away")
    - market_price            (American odds for the chosen side)
    - book_spread_home        (home team spread at that book)
    - model_fair_spread       (our fair spread for the home team; blended)
    - model_edge_pts          (edge in points for the chosen side)
    - book_implied_prob       (book's implied probability from the price)
    - model_side_prob         (our estimated win/cover probability)
    - suggested_kelly         (Kelly fraction of bankroll)
    - suggested_stake_units   (stake in abstract "units")
    - consensus_close         (consensus closing home spread across books)
    - book_dispersion         (std dev of home spreads across books)

The selection logic is conservative but configurable via the constants in
the CONFIG section below.
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUTPUTS_DIR = "outputs"
SNAPSHOT_DIR = "data/_snapshots"
SNAPSHOT_TYPE = "close"

# Edge thresholds
MIN_EDGE_PTS = 1.5        # minimum edge in points on the chosen side
MIN_MODEL_PROB = 0.55     # minimum model spread-cover probability
MIN_KELLY = 0.01          # minimum Kelly fraction to consider a bet
MAX_KELLY = 0.07          # cap Kelly for sanity

# Kelly -> units translation
BANKROLL_UNITS = 100.0    # "bankroll" measured in abstract units
MAX_UNITS = 3.0           # never bet more than this many units on a single play

# Mapping from point edge to cover probability via logistic curve.
# edge_pts = book_spread_home - fair_spread (for home side).
# For the chosen side, we use edge_pts >= 0 and:
#     P(cover) = sigmoid(ALPHA * edge_pts)
COVER_PROB_ALPHA = 0.25   # slope of the logistic; tune via backtests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_prob(odds: float) -> Optional[float]:
    """
    Convert American odds to implied probability (single-sided, no-vig).

    Returns None if odds is missing or invalid.
    """
    if odds is None:
        return None

    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None

    if o == 0.0 or math.isnan(o):
        return None

    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return -o / (-o + 100.0)


def cover_prob_from_edge(edge_pts: float, alpha: float = COVER_PROB_ALPHA) -> float:
    """
    Map a point edge (>= 0) to an approximate spread-cover probability
    using a logistic curve:

        P(cover) = sigmoid(alpha * edge_pts)

    where:
        edge_pts = book_spread_home - fair_spread
    for the chosen side.

    This is a simple, monotonic approximation that can be calibrated later
    with historical data.
    """
    if edge_pts <= 0 or math.isnan(edge_pts):
        return 0.5
    z = alpha * edge_pts
    return 1.0 / (1.0 + math.exp(-z))


def _find_latest_predictions_file(outputs_dir: str = OUTPUTS_DIR) -> Optional[str]:
    """
    Find the latest predictions_*_market.csv file in the outputs directory.

    Files are expected to look like:
        predictions_YYYY-MM-DD_market.csv
    """
    pattern = os.path.join(outputs_dir, "predictions_*_market.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort()
    latest = candidates[-1]
    return latest


def _extract_run_date_from_predictions_path(path: str) -> str:
    """
    Given a path like:
        outputs/predictions_2025-12-11_market.csv
    return:
        "2025-12-11"
    """
    base = os.path.basename(path)
    # predictions_YYYY-MM-DD_market.csv
    without_prefix = base.replace("predictions_", "")
    date_part = without_prefix.split("_market")[0]
    return date_part


def _find_close_snapshot_for_date(
    run_date: str,
    snapshot_dir: str = SNAPSHOT_DIR,
    snapshot_type: str = SNAPSHOT_TYPE,
) -> Optional[str]:
    """
    Find the latest CLOSE snapshot CSV for a given run_date.

    Snapshots are expected to look like:
        {snapshot_type}_YYYYMMDD_HHMMSS.csv
    """
    ymd = run_date.replace("-", "")
    pattern = os.path.join(snapshot_dir, f"{snapshot_type}_{ymd}_*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


@dataclass
class PickConfig:
    min_edge_pts: float = MIN_EDGE_PTS
    min_model_prob: float = MIN_MODEL_PROB
    min_kelly: float = MIN_KELLY
    max_kelly: float = MAX_KELLY
    alpha: float = COVER_PROB_ALPHA
    bankroll_units: float = BANKROLL_UNITS
    max_units: float = MAX_UNITS


# ---------------------------------------------------------------------------
# Core pick generation
# ---------------------------------------------------------------------------

def generate_spread_picks(
    preds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    config: PickConfig,
) -> pd.DataFrame:
    """
    Generate spread picks by comparing our fair spread to each book's line.

    For each game/book, we:
        - compute home_edge_pts = book_spread_home - fair_spread
        - compute away_edge_pts = -home_edge_pts
        - choose the side (home/away) with the larger positive edge
        - map edge_pts -> model_side_prob via logistic
        - compare model_side_prob to book implied probability from the price
        - compute Kelly fraction
        - filter by thresholds in config
    We then keep at most ONE pick per game (the highest-Kelly candidate).
    """
    if preds_df.empty or odds_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "book",
                "market_side",
                "market_price",
                "book_spread_home",
                "model_fair_spread",
                "model_edge_pts",
                "book_implied_prob",
                "model_side_prob",
                "suggested_kelly",
                "suggested_stake_units",
                "consensus_close",
                "book_dispersion",
            ]
        )

    # Merge predictions with CLOSE snapshot odds on merge_key + teams + date
    merge_cols = ["merge_key", "game_date", "home_team", "away_team"]
    missing_cols = [c for c in merge_cols if c not in preds_df.columns or c not in odds_df.columns]
    if missing_cols:
        raise ValueError(f"Missing merge columns in preds/odds: {missing_cols}")

    merged = odds_df.merge(
        preds_df,
        on=merge_cols,
        how="inner",
        suffixes=("", "_pred"),
    )

    if merged.empty:
        logger.warning("[edge_picker] No rows after merging preds and odds.")
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "book",
                "market_side",
                "market_price",
                "book_spread_home",
                "model_fair_spread",
                "model_edge_pts",
                "book_implied_prob",
                "model_side_prob",
                "suggested_kelly",
                "suggested_stake_units",
                "consensus_close",
                "book_dispersion",
            ]
        )

    # Ensure numeric types for spread/price columns
    for col in [
        "spread_home_point",
        "spread_away_point",
        "spread_home_price",
        "spread_away_price",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    candidates: List[dict] = []

    for _, row in merged.iterrows():
        line_home = row.get("spread_home_point", np.nan)
        price_home = row.get("spread_home_price", np.nan)
        price_away = row.get("spread_away_price", np.nan)
        fair_spread = row.get("fair_spread", np.nan)  # blended fair spread for home

        if pd.isna(line_home) or pd.isna(fair_spread):
            continue

        # Edge in points from home perspective
        home_edge_pts = line_home - fair_spread
        away_edge_pts = -home_edge_pts

        best_side: Optional[str] = None
        edge_pts: float = 0.0
        market_price: Optional[float] = None

        # Decide which side (if any) has enough edge in points
        if home_edge_pts >= config.min_edge_pts and home_edge_pts >= away_edge_pts:
            best_side = "home"
            edge_pts = float(home_edge_pts)
            market_price = price_home
        elif away_edge_pts >= config.min_edge_pts and away_edge_pts > home_edge_pts:
            best_side = "away"
            edge_pts = float(away_edge_pts)
            market_price = price_away
        else:
            continue  # no strong edge on either side

        if best_side is None or market_price is None or pd.isna(market_price):
            continue

        # Model's estimated cover probability for the chosen side
        model_side_prob = cover_prob_from_edge(edge_pts, alpha=config.alpha)
        if model_side_prob < config.min_model_prob:
            continue

        # Book's implied probability from the chosen side's price
        book_implied_prob = american_to_prob(market_price)
        if book_implied_prob is None:
            continue

        # Compute Kelly fraction: k = (p*(b+1) - 1) / b, where b = decimal_odds - 1
        if market_price > 0:
            decimal_odds = 1.0 + market_price / 100.0
        else:
            decimal_odds = 1.0 + 100.0 / abs(market_price)

        b = decimal_odds - 1.0
        if b <= 0:
            continue

        kelly = (model_side_prob * (b + 1.0) - 1.0) / b
        if kelly <= config.min_kelly:
            continue

        kelly = float(min(kelly, config.max_kelly))

        # Translate Kelly fraction into "units"
        suggested_units = min(config.max_units, config.bankroll_units * kelly)

        if suggested_units <= 0:
            continue

        candidates.append(
            {
                "game_id": row.get("game_id"),
                "game_date": row.get("game_date"),
                "book": row.get("book"),
                "market_side": best_side,
                "market_price": market_price,
                "book_spread_home": line_home,
                "model_fair_spread": fair_spread,
                "model_edge_pts": edge_pts,
                "book_implied_prob": book_implied_prob,
                "model_side_prob": model_side_prob,
                "suggested_kelly": kelly,
                "suggested_stake_units": suggested_units,
                "consensus_close": row.get("consensus_close", np.nan),
                "book_dispersion": row.get("book_dispersion", np.nan),
            }
        )

    if not candidates:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "book",
                "market_side",
                "market_price",
                "book_spread_home",
                "model_fair_spread",
                "model_edge_pts",
                "book_implied_prob",
                "model_side_prob",
                "suggested_kelly",
                "suggested_stake_units",
                "consensus_close",
                "book_dispersion",
            ]
        )

    # From the candidate list, select the best (highest Kelly) per game_id
    candidates_df = pd.DataFrame(candidates)
    picks = (
        candidates_df.sort_values("suggested_kelly", ascending=False)
        .groupby("game_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # Round for readability
    for col in [
        "book_spread_home",
        "model_fair_spread",
        "model_edge_pts",
        "book_implied_prob",
        "model_side_prob",
        "suggested_kelly",
        "suggested_stake_units",
        "consensus_close",
        "book_dispersion",
    ]:
        if col in picks.columns:
            picks[col] = picks[col].astype(float).round(4)

    return picks[
        [
            "game_id",
            "game_date",
            "book",
            "market_side",
            "market_price",
            "book_spread_home",
            "model_fair_spread",
            "model_edge_pts",
            "book_implied_prob",
            "model_side_prob",
            "suggested_kelly",
            "suggested_stake_units",
            "consensus_close",
            "book_dispersion",
        ]
    ]


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------

def main(run_date: Optional[str] = None) -> None:
    """
    Entry point used by run_daily.py and CLI.

    Steps:
        1) Discover latest predictions_*_market.csv if run_date is not provided.
        2) Resolve run_date from that filename.
        3) Find CLOSE snapshot CSV for that date.
        4) Generate spread picks.
        5) Write outputs/picks_YYYY-MM-DD.csv and picks_report.html.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [edge_picker] %(message)s",
    )

    # 1) Resolve predictions file & run date
    predictions_path = _find_latest_predictions_file()
    if predictions_path is None:
        logger.warning("No predictions_*_market.csv files found in %s; no picks generated.", OUTPUTS_DIR)
        return

    inferred_date = _extract_run_date_from_predictions_path(predictions_path)
    if run_date is None:
        run_date = inferred_date
    else:
        # If run_date is provided, override the predictions_path if a better match exists
        candidate_path = os.path.join(OUTPUTS_DIR, f"predictions_{run_date}_market.csv")
        if os.path.exists(candidate_path):
            predictions_path = candidate_path

    logger.info("Using predictions file: %s (run_date=%s)", predictions_path, run_date)

    preds = pd.read_csv(predictions_path)

    # 2) Find CLOSE snapshot for run_date
    snapshot_path = _find_close_snapshot_for_date(run_date)
    if snapshot_path is None:
        logger.warning(
            "No CLOSE snapshot found for %s in %s; picks will not be generated.",
            run_date,
            SNAPSHOT_DIR,
        )
        picks = pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "book",
                "market_side",
                "market_price",
                "book_spread_home",
                "model_fair_spread",
                "model_edge_pts",
                "book_implied_prob",
                "model_side_prob",
                "suggested_kelly",
                "suggested_stake_units",
                "consensus_close",
                "book_dispersion",
            ]
        )
    else:
        logger.info("Using CLOSE snapshot: %s", snapshot_path)
        odds = pd.read_csv(snapshot_path)

        cfg = PickConfig()
        picks = generate_spread_picks(preds, odds, cfg)

    # 3) Write picks CSV
    picks_csv_path = os.path.join(OUTPUTS_DIR, f"picks_{run_date}.csv")
    picks.to_csv(picks_csv_path, index=False)
    logger.info("Wrote picks CSV to %s (%d rows).", picks_csv_path, len(picks))

    # 4) Write simple HTML report
    html_path = "picks_report.html"
    if picks.empty:
        html_content = f"<html><body><h1>Picks for {run_date}</h1><p>No edges today.</p></body></html>"
    else:
        # Basic HTML table for quick viewing
        html_content = [
            "<html><body>",
            f"<h1>Picks for {run_date}</h1>",
            "<table border='1' cellpadding='4' cellspacing='0'>",
            "<tr>",
        ]
        for col in picks.columns:
            html_content.append(f"<th>{col}</th>")
        html_content.append("</tr>")

        for _, row in picks.iterrows():
            html_content.append("<tr>")
            for col in picks.columns:
                html_content.append(f"<td>{row[col]}</td>")
            html_content.append("</tr>")

        html_content.append("</table></body></html>")
        html_content = "".join(html_content)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Wrote picks HTML report to %s.", html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NBA spread picks from predictions and odds.")
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Run date in YYYY-MM-DD (defaults to latest predictions_*_market.csv date).",
    )
    args = parser.parse_args()
    main(run_date=args.run_date)
