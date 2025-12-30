"""
ATS Execution Stress Test (E4.1)
================================

Purpose
-------
Quantify how ATS edge holds up under more realistic execution assumptions by
recomputing ROI/CLV/drawdown under multiple execution-price modes.

This script is deliberately *selection-locked*: it assumes the input CSV already
contains the set of bets you want to stress test (same games), and it only varies
the execution spread used to settle those bets.

Inputs
------
CSV passed as positional arg (e.g., outputs/ats_roi_input.csv) must contain:
- home_final_score (numeric)
- away_final_score (numeric)
- home_spread_consensus_open (numeric)
- home_spread_consensus_close (numeric)

Optional (improves WORST_BOOK_CLOSE approximation):
- close_dispersion OR home_spread_dispersion_close OR home_spread_dispersion

Outputs
-------
- Prints a table to stdout
- Writes a CSV (default outputs/ats_execution_stress_test.csv)

Conventions
-----------
- Spread is represented as the *home spread* (home perspective).
- We assume the bet side is AWAY (ATS v1 away_only). Away covers if:
      home_score + home_spread < away_score
  i.e., away margin is better than the home line.

- Profit at -110 for 1u stake:
      win  -> +100/110
      loss -> -1
      push -> 0

Execution Modes
---------------
CONSENSUS_CLOSE
CONSENSUS_OPEN
WORST_BOOK_CLOSE            (approx: close_consensus + abs(dispersion))
CONSENSUS_CLOSE_PLUS_0.25   (worse for away)
CONSENSUS_CLOSE_PLUS_0.5    (worse for away)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


PPU_ATS_MINUS_110 = 100.0 / 110.0  # profit per 1u stake at -110


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _get_close_dispersion_row(row: pd.Series) -> float:
    """
    Try multiple column names for close dispersion.
    Returns 0.0 if not available.
    """
    for c in ("close_dispersion", "home_spread_dispersion_close", "home_spread_dispersion"):
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                return 0.0
    return 0.0


def _settle_away(home_score: float, away_score: float, home_spread: float) -> str:
    """
    Settle an AWAY ATS bet using home spread (home perspective).
    Away covers if adjusted home score is strictly less than away score.
    """
    adj_home = home_score + home_spread
    if adj_home == away_score:
        return "push"
    return "win" if adj_home < away_score else "loss"


# -----------------------------------------------------------------------------
# Execution price selectors
# (Return executed HOME spread line.)
#
# NOTE ON SLIPPAGE DIRECTION:
# For an AWAY bet, a worse fill means the home spread moves UP (toward zero / more positive),
# reducing the points the away side receives.
# -----------------------------------------------------------------------------
def consensus_close(row: pd.Series) -> float:
    return float(row["home_spread_consensus_close"])


def consensus_open(row: pd.Series) -> float:
    return float(row["home_spread_consensus_open"])


def worst_book_close(row: pd.Series) -> float:
    # Approximate worst/highest home line among books:
    # worst ≈ mean + |std|
    base = float(row["home_spread_consensus_close"])
    disp = abs(_get_close_dispersion_row(row))
    return base + disp


def consensus_close_plus_025(row: pd.Series) -> float:
    return float(row["home_spread_consensus_close"]) + 0.25


def consensus_close_plus_050(row: pd.Series) -> float:
    return float(row["home_spread_consensus_close"]) + 0.50


EXECUTION_SELECTORS: Dict[str, Callable[[pd.Series], float]] = {
    "CONSENSUS_CLOSE": consensus_close,
    "CONSENSUS_OPEN": consensus_open,
    "WORST_BOOK_CLOSE": worst_book_close,
    "CONSENSUS_CLOSE_PLUS_0.25": consensus_close_plus_025,
    "CONSENSUS_CLOSE_PLUS_0.5": consensus_close_plus_050,
}


@dataclass(frozen=True)
class ExecutionMetrics:
    mode: str
    bets: int
    roi: float
    clv: float
    win_rate: float
    max_drawdown: float


def compute_metrics_for_mode(df: pd.DataFrame, selector_name: str) -> ExecutionMetrics:
    if selector_name not in EXECUTION_SELECTORS:
        raise ValueError(f"Unknown execution mode: {selector_name}")

    selector = EXECUTION_SELECTORS[selector_name]

    # Executed spreads (home line) for this mode
    executed_spreads = df.apply(selector, axis=1).astype(float)

    # Closing spreads for CLV (close - executed)
    closing_spreads = _to_float_series(df["home_spread_consensus_close"]).astype(float)

    # Scores
    home_score = _to_float_series(df["home_final_score"]).astype(float)
    away_score = _to_float_series(df["away_final_score"]).astype(float)

    # Settle each bet (AWAY ATS)
    results = []
    profits = []
    for hs, aw, line in zip(home_score, away_score, executed_spreads):
        if pd.isna(hs) or pd.isna(aw) or pd.isna(line):
            # should not happen if inputs are clean; treat as no-action
            results.append("no_bet")
            profits.append(0.0)
            continue

        r = _settle_away(float(hs), float(aw), float(line))
        results.append(r)
        if r == "win":
            profits.append(PPU_ATS_MINUS_110)
        elif r == "loss":
            profits.append(-1.0)
        else:
            profits.append(0.0)

    profits_arr = np.array(profits, dtype=float)
    bets = int(len(profits_arr))

    # ROI per bet (units)
    cum = np.cumsum(profits_arr) if bets else np.array([0.0])
    roi = float(cum[-1] / bets) if bets else 0.0

    # Win rate (exclude pushes from numerator by definition? keep simple: wins / bets)
    win_rate = float(np.mean(np.array(results) == "win")) if bets else 0.0

    # CLV: close - executed (negative if you paid worse than close)
    clv = float(np.nanmean(closing_spreads - executed_spreads))

    # Max drawdown
    running_max = np.maximum.accumulate(cum) if bets else np.array([0.0])
    dd = cum - running_max
    max_drawdown = float(np.min(dd)) if bets else 0.0

    return ExecutionMetrics(
        mode=selector_name,
        bets=bets,
        roi=roi,
        clv=clv,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
    )


def run_stress_test(roi_input_path: str) -> pd.DataFrame:
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

    # Coerce key numeric columns once (keeps selectors stable)
    for c in list(required_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    metrics: List[ExecutionMetrics] = []
    for mode in EXECUTION_SELECTORS.keys():
        metrics.append(compute_metrics_for_mode(df, mode))

    result_df = pd.DataFrame(
        [
            {
                "execution_mode": m.mode,
                "bets": m.bets,
                "roi": m.roi,
                "clv": m.clv,
                "win_rate": m.win_rate,
                "max_drawdown": m.max_drawdown,
            }
            for m in metrics
        ]
    )

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test ATS ROI under various execution price assumptions (E4.1)."
    )
    parser.add_argument(
        "roi_input",
        type=str,
        help="Path to the ATS ROI input CSV (e.g., outputs/ats_roi_input.csv).",
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
