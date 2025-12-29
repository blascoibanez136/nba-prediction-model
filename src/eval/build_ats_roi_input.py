"""
Utility script to build the per‑game input for ATS ROI analysis with market features.

This script merges the backtest joined file (model predictions and game results)
with consensus spread and dispersion statistics computed from the odds snapshots.

The resulting CSV contains the original per‑game columns plus two new columns:

    home_spread_consensus: average of spread_home_point across books (close snapshot)
    home_spread_dispersion: standard deviation of spread_home_point across books

If a game has no snapshot data, the consensus/dispersion fields will be null.

Usage:

    python -m src.eval.build_ats_roi_input \
        --backtest-joined outputs/backtest_joined.csv \
        --snapshot-dir data/_snapshots \
        --start 2023-10-24 \
        --end 2024-04-14 \
        --out outputs/backtest_joined_market.csv

The start and end dates bound the snapshot search.  The script looks for
files in snapshot_dir matching pattern "close_YYYYMMDD*.json" for each date
between start and end (inclusive).  It loads the first matching JSON file per
day, computes dispersion via src.ingest.odds_snapshots.compute_dispersion,
and concatenates the results.

Determinism: file order and merge keys are sorted by merge_key for stable
output.  Missing consensus or dispersion values propagate as NaN.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd

from src.ingest.odds_snapshots import compute_dispersion, _norm


def _iter_snapshot_files(snapshot_dir: Path, start: str, end: str) -> Iterable[Path]:
    """Yield close snapshot JSON paths between start and end (inclusive).

    For each date in the range, looks for files matching 'close_YYYYMMDD*.json'.
    Picks the lexicographically first file if multiple exist.
    """
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    for date in pd.date_range(start_date, end_date, freq="D"):
        date_str = date.strftime("%Y%m%d")
        pattern = f"close_{date_str}"
        candidates = sorted(
            [p for p in snapshot_dir.glob(f"{pattern}*.json") if p.is_file()]
        )
        if candidates:
            yield candidates[0]


def build_market_df(snapshot_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Compute consensus and dispersion stats for the date range.

    Returns a DataFrame with columns [merge_key, home_spread_consensus, home_spread_dispersion].
    """
    frames = []
    for path in _iter_snapshot_files(snapshot_dir, start, end):
        try:
            df = compute_dispersion(path)
        except Exception as e:
            print(f"[build_ats_roi_input] WARNING: failed to compute dispersion for {path}: {e}")
            continue
        # compute_dispersion returns columns: merge_key, consensus_close, book_dispersion
        # We rename them to match ats_roi_analysis expectations.
        df = df.rename(
            columns={
                "consensus_close": "home_spread_consensus",
                "book_dispersion": "home_spread_dispersion",
            }
        )[["merge_key", "home_spread_consensus", "home_spread_dispersion"]]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])
    market_df = pd.concat(frames, ignore_index=True)
    # drop duplicates: keep last (latest date) per merge_key
    market_df = market_df.sort_values("merge_key").drop_duplicates("merge_key", keep="last")
    return market_df


def attach_market_features(backtest_path: Path, market_df: pd.DataFrame) -> pd.DataFrame:
    """Attach consensus and dispersion to the backtest per‑game DataFrame.

    Computes merge_key from home_team, away_team, and the date column (game_date/date/gamedate)
    and left‑joins the market_df on merge_key.  If none of the expected date columns are
    present, raises a RuntimeError.
    """
    df = pd.read_csv(backtest_path)
    # Find the date column (match ats_roi_analysis logic)
    date_col = None
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise RuntimeError(
            "[build_ats_roi_input] Missing game date column (expected one of game_date, date, gamedate)"
        )
    # Normalize team names to canonical form and build merge_key
    def build_key(row) -> str:
        return f"{_norm(row['home_team'])}__{_norm(row['away_team'])}__{row[date_col]}"

    df["merge_key"] = df.apply(build_key, axis=1)
    # Left join market features
    merged = df.merge(market_df, on="merge_key", how="left")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest-joined", type=str, required=True, help="Path to backtest_joined.csv")
    ap.add_argument("--snapshot-dir", type=str, required=True, help="Directory containing odds snapshots")
    ap.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = ap.parse_args()

    backtest_path = Path(args.backtest_joined)
    snapshot_dir = Path(args.snapshot_dir)
    out_path = Path(args.out)

    market_df = build_market_df(snapshot_dir, args.start, args.end)
    merged_df = attach_market_features(backtest_path, market_df)
    merged_df.to_csv(out_path, index=False)
    # Print coverage metrics for logging
    consensus_cov = merged_df["home_spread_consensus"].notna().mean() * 100
    disp_cov = merged_df["home_spread_dispersion"].notna().mean() * 100
    print(
        f"[build_ats_roi_input] wrote: {out_path} (consensus_cov={consensus_cov:.1f}% dispersion_cov={disp_cov:.1f}%)"
    )


if __name__ == "__main__":
    main()
