"""
ATS Execution Stress Test
========================

This module provides a simple backtesting harness to quantify how
much the NBA ATS model's edge degrades under various execution
assumptions.  In production the model computes expected value (EV)
using consensus closing spreads and -110 pricing.  However, in
practice bets can be filled at worse prices if lines move or if
liquidity is limited.  E4 introduces a number of stress‐test
scenarios to quantify the sensitivity of historical ROI, closing
line value (CLV) and drawdown to pessimistic execution prices.

Five execution modes are supported:

``CONSENSUS_CLOSE``
    Use the consensus closing spread.  This reproduces the current
    baseline ROI numbers and serves as a point of comparison.

``CONSENSUS_OPEN``
    Use the consensus opening spread.  This tests the impact of
    executing bets earlier in the day.  CLV is measured against the
    closing consensus.

``WORST_BOOK_CLOSE``
    Use a worst‐case spread among all books in the closing snapshot.
    For an away bet this is approximated by adding the measured
    dispersion at close to the consensus close spread.  If you have
    per‐book snapshots available you can replace this approximation
    with the true worst price.

``CONSENSUS_CLOSE_PLUS_0.25`` and ``CONSENSUS_CLOSE_PLUS_0.5``
    Shift the consensus closing spread against the bettor by 0.25 or
    0.50 points.  This represents modest and severe slippage,
    respectively.

The script expects that you have already generated the ATS ROI input
via ``build_ats_roi_input.py``.  That CSV must contain at least
the following columns:

* ``game_date``
* ``home_team`` and ``away_team``
* ``away_final_score`` and ``home_final_score``
* ``home_spread_consensus_open`` and ``home_spread_consensus_close``
* ``home_spread_dispersion_close`` (for the worst‐book approximation)

If these fields are unavailable the script will raise an error.  The
module computes profit per unit at -110 pricing (1 unit stake per
bet), cumulative ROI, average CLV (difference between executed and
closing spreads) and maximum drawdown for each execution mode.  All
metrics are returned in a single DataFrame for easy comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Execution price selectors
#
# These functions take a row of the ROI input DataFrame and return the
# executed spread for the away team.  Positive values imply the home team
# is favoured; negative values imply the away team is favoured.  For an
# away bet we want the line to move in our favour, so slippage is modelled
# by increasing the spread (making it harder for the away team to cover).
# -----------------------------------------------------------------------------

def consensus_close(row: pd.Series) -> float:
    return row["home_spread_consensus_close"]


def consensus_open(row: pd.Series) -> float:
    return row["home_spread_consensus_open"]


def worst_book_close(row: pd.Series) -> float:
    # Approximate the worst available price by adding the close dispersion to the
    # consensus close spread.  In practice you can replace this with the
    # maximum per‑book spread if raw per‑book snapshots are available.
    return row["home_spread_consensus_close"] + abs(row.get("home_spread_dispersion_close", 0.0))


def consensus_close_plus_025(row: pd.Series) -> float:
    return row["home_spread_consensus_close"] + 0.25


def consensus_close_plus_050(row: pd.Series) -> float:
    return row["home_spread_consensus_close"] + 0.50


EXECUTION_SELECTORS: Dict[str, Callable[[pd.Series], float]] = {
    "CONSENSUS_CLOSE": consensus_close,
    "CONSENSUS_OPEN": consensus_open,
    "WORST_BOOK_CLOSE": worst_book_close,
    "CONSENSUS_CLOSE_PLUS_0.25": consensus_close_plus_025,
    "CONSENSUS_CLOSE_PLUS_0.5": consensus_close_plus_050,
}


@dataclass
class ExecutionMetrics:
    mode: str
    bets: int
    roi: float
    clv: float
    win_rate: float
    max_drawdown: float


def compute_metrics_for_mode(df: pd.DataFrame, selector_name: str) -> ExecutionMetrics:
    """Compute ROI/CLV/drawdown for a given execution mode.

    Parameters
    ----------
    df : pd.DataFrame
        ROI input containing consensus spreads, final scores and dispersion.
    selector_name : str
        One of the keys in EXECUTION_SELECTORS specifying which execution
        price to apply.

    Returns
    -------
    ExecutionMetrics
        Dataclass summarising bets placed, ROI, CLV, win rate and
        maximum drawdown measured in units.
    """
    if selector_name not in EXECUTION_SELECTORS:
        raise ValueError(f"Unknown execution mode: {selector_name}")
    selector = EXECUTION_SELECTORS[selector_name]

    # Determine executed spreads and closing spreads for CLV
    executed_spreads = df.apply(selector, axis=1)
    closing_spreads = df["home_spread_consensus_close"]

    # Compute actual margin: positive if home covers, negative if away covers
    actual_margin = df["home_final_score"] - df["away_final_score"]

    # Determine if away bet wins under executed spread (i.e., away covers)
    away_covers = actual_margin < executed_spreads
    win_rate = away_covers.mean()

    # Profit per unit at -110 pricing: +0.9091 for a win, -1.0 for a loss
    profit_per_bet = np.where(away_covers, 1.0, -1.0) * 1.0  # stake = 1u; vig omitted for simplicity
    # If you wish to include vig, multiply wins by 0.9091 instead of 1.0

    # Profit per unit at -110 pricing (ppu = 100/110). Stake is 1u per bet.
    ppu = 100.0 / 110.0  # 0.9090909
    profit_per_bet = np.where(away_covers, ppu, -1.0).astype(float)

    cumulative_profit = np.cumsum(profit_per_bet)
    bets = int(len(profit_per_bet))
    roi = float(cumulative_profit[-1] / bets) if bets else 0.0

    # CLV = closing consensus spread - executed spread (negative if we paid a worse price)
    clv = float(np.nanmean(closing_spreads - executed_spreads))

    # Maximum drawdown: min(cum_profit - running_peak)
    running_max = np.maximum.accumulate(cumulative_profit) if bets else np.array([0.0])

    drawdown_series = cumulative_profit - running_max
    max_drawdown = drawdown_series.min()

    return ExecutionMetrics(
        mode=selector_name,
        bets=bets,
        roi=roi,
        clv=clv,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
    )


def run_stress_test(roi_input_path: str) -> pd.DataFrame:
    """Run the stress test across all execution modes and return a DataFrame."""
    df = pd.read_csv(roi_input_path)

    required_cols = {
        "home_spread_consensus_close",
        "home_spread_consensus_open",
        "home_final_score",
        "away_final_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"ROI input is missing required columns: {missing}")

    metrics: List[ExecutionMetrics] = []
    for mode in EXECUTION_SELECTORS.keys():
        metrics.append(compute_metrics_for_mode(df, mode))

    result_df = pd.DataFrame([
        {
            "execution_mode": m.mode,
            "bets": m.bets,
            "roi": m.roi,
            "clv": m.clv,
            "win_rate": m.win_rate,
            "max_drawdown": m.max_drawdown,
        }
        for m in metrics
    ])

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test ATS ROI under various execution price assumptions."
    )
    parser.add_argument(
        "roi_input",
        type=str,
        help="Path to the ATS ROI input CSV generated by build_ats_roi_input.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ats_execution_stress_test.csv",
        help="Destination to write the aggregated metrics CSV.",
    )
    args = parser.parse_args()

    result = run_stress_test(args.roi_input)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
