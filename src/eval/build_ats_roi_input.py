"""
Utility script to build the per-game input for ATS ROI analysis with market features.

This script merges the backtest joined file (model predictions and game results)
with consensus spread and dispersion computed from per-book odds snapshots.

It supports BOTH:
  - normalized CSV snapshots: close_YYYYMMDD.csv (recommended / what you have)
  - raw JSON snapshots: close_YYYYMMDD*.json (legacy support)

Output adds:
  - home_spread_consensus: mean(spread_home_point) across books
  - home_spread_dispersion: std(spread_home_point) across books

Example:
python -m src.eval.build_ats_roi_input \
    --backtest-joined outputs/backtest_joined.csv \
    --snapshot-dir data/_snapshots \
    --start 2023-10-24 \
    --end 2024-04-14 \
    --out outputs/backtest_joined_market.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.ingest.odds_snapshots import compute_dispersion, _norm


# -----------------------
# Snapshot discovery
# -----------------------

def _iter_close_snapshot_files(snapshot_dir: Path, start: str, end: str) -> Iterable[Path]:
    """
    Yield a close snapshot path per date in [start, end], preferring CSV.

    Preferred:
      close_YYYYMMDD.csv

    Fallback:
      close_YYYYMMDD*.json  (if present)
    """
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    for d in pd.date_range(start_date, end_date, freq="D"):
        ymd = d.strftime("%Y%m%d")

        # 1) Prefer normalized CSV snapshots (what you have)
        csv_path = snapshot_dir / f"close_{ymd}.csv"
        if csv_path.exists():
            yield csv_path
            continue

        # 2) Fallback to JSON snapshots (if any)
        json_candidates = sorted([p for p in snapshot_dir.glob(f"close_{ymd}*.json") if p.is_file()])
        if json_candidates:
            yield json_candidates[0]
            continue

        # else: no snapshot for that day


# -----------------------
# Market feature builder
# -----------------------

def _market_from_close_csv(path: Path) -> pd.DataFrame:
    """
    Compute consensus + dispersion from a normalized close CSV snapshot.
    Requires columns: merge_key, spread_home_point (book optional but typical).
    """
    df = pd.read_csv(path)
    if df.empty or "merge_key" not in df.columns or "spread_home_point" not in df.columns:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    df = df.copy()
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    df["spread_home_point"] = pd.to_numeric(df["spread_home_point"], errors="coerce")
    df = df.dropna(subset=["merge_key", "spread_home_point"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    out = (
        df.groupby("merge_key")["spread_home_point"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
        .reset_index()
    )
    return out


def build_market_df(snapshot_dir: Path, start: str, end: str) -> pd.DataFrame:
    """
    Build a per-merge_key market feature table over the date range.

    Output columns:
      merge_key, home_spread_consensus, home_spread_dispersion
    """
    frames = []

    for path in _iter_close_snapshot_files(snapshot_dir, start, end):
        try:
            if path.suffix.lower() == ".csv":
                m = _market_from_close_csv(path)
            else:
                # JSON fallback via existing helper (returns consensus_close + book_dispersion)
                m = compute_dispersion(path).rename(
                    columns={
                        "consensus_close": "home_spread_consensus",
                        "book_dispersion": "home_spread_dispersion",
                    }
                )[["merge_key", "home_spread_consensus", "home_spread_dispersion"]]
            frames.append(m)
        except Exception as e:
            print(f"[build_ats_roi_input] WARNING: failed on {path.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    market_df = pd.concat(frames, ignore_index=True)
    market_df["merge_key"] = market_df["merge_key"].astype(str).str.strip().str.lower()
    market_df = market_df.sort_values("merge_key").drop_duplicates("merge_key", keep="last")
    return market_df


# -----------------------
# Merge into backtest_joined
# -----------------------

def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[build_ats_roi_input] No date column found (expected game_date/date/gamedate).")


def attach_market_features(backtest_path: Path, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach consensus and dispersion to backtest_joined, keyed by:
      merge_key = home__away__YYYY-MM-DD   (lowercase)
    """
    df = pd.read_csv(backtest_path)
    if df.empty:
        raise RuntimeError("[build_ats_roi_input] backtest_joined is empty")

    for c in ["home_team", "away_team"]:
        if c not in df.columns:
            raise RuntimeError(f"[build_ats_roi_input] Missing required column: {c}")

    date_col = _detect_date_col(df)
    # normalize date to YYYY-MM-DD string
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise RuntimeError("[build_ats_roi_input] Found NaT in date column; cannot build merge_key safely.")
    df["_game_date_str"] = df[date_col].dt.strftime("%Y-%m-%d")

    def mk(row) -> str:
        return f"{_norm(row['home_team'])}__{_norm(row['away_team'])}__{row['_game_date_str']}"

    df["merge_key"] = df.apply(mk, axis=1)
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()

    merged = df.merge(market_df, on="merge_key", how="left")
    merged = merged.drop(columns=["_game_date_str"], errors="ignore")
    return merged


# -----------------------
# CLI
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest-joined", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    backtest_path = Path(args.backtest_joined)
    out_path = Path(args.out)

    market_df = build_market_df(snapshot_dir, args.start, args.end)
    merged = attach_market_features(backtest_path, market_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    cov_consensus = float(merged["home_spread_consensus"].notna().mean() * 100.0) if "home_spread_consensus" in merged.columns else 0.0
    cov_disp = float(merged["home_spread_dispersion"].notna().mean() * 100.0) if "home_spread_dispersion" in merged.columns else 0.0
    print(f"[build_ats_roi_input] wrote: {out_path} (consensus_cov={cov_consensus:.1f}% dispersion_cov={cov_disp:.1f}%)")


if __name__ == "__main__":
    main()
