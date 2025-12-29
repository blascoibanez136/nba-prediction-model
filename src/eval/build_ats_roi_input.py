"""
Utility script to build the per-game input for ATS ROI analysis with market features.

This script merges the backtest joined file (model predictions and game results)
with consensus spread and dispersion computed from odds snapshots.

It supports BOTH snapshot formats commonly found in this repo:

A) Normalized per-book CSV snapshots (preferred)
   - close_YYYYMMDD.csv   (e.g. close_20240329.csv)

   Required columns in the snapshot CSV:
     - merge_key
     - spread_home_point

B) Raw Odds API JSON snapshots (fallback)
   - close_YYYYMMDD*.json
   - raw_YYYY-MM-DD*.json  (e.g. raw_2023-10-24.json)
   - raw_close_YYYY-MM-DD*.json (if present)

For JSON, we use src.ingest.odds_snapshots.compute_dispersion(), which returns:
  merge_key, consensus_close, book_dispersion

Output adds these columns to the backtest file:
  - home_spread_consensus  (mean of spread_home_point across books)
  - home_spread_dispersion (std dev of spread_home_point across books)

If a game has no snapshot data, the consensus/dispersion fields will be NaN.

Usage:

python -m src.eval.build_ats_roi_input \
  --backtest-joined outputs/backtest_joined.csv \
  --snapshot-dir data/_snapshots \
  --start 2023-10-24 \
  --end 2024-04-14 \
  --out outputs/backtest_joined_market.csv

Notes:
- If your snapshot dir contains a "raw/" subfolder, this script will search it too.
- Deterministic: per-day file selection is stable (lexicographically first match).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.ingest.odds_snapshots import compute_dispersion, _norm


# -----------------------------
# Snapshot discovery
# -----------------------------

def _candidate_snapshot_dirs(snapshot_dir: Path) -> List[Path]:
    """Return snapshot_dir plus common subfolders (e.g. raw/) if they exist."""
    dirs = [snapshot_dir]
    raw = snapshot_dir / "raw"
    if raw.exists() and raw.is_dir():
        dirs.append(raw)
    return dirs


def _pick_first(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p is not None and p.exists() and p.is_file()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[0]


def _iter_snapshot_files(snapshot_dir: Path, start: str, end: str) -> Iterable[Path]:
    """
    Yield ONE close snapshot per date in [start, end] inclusive.

    Priority per date:
      1) close_YYYYMMDD.csv
      2) close_YYYYMMDD*.json
      3) raw_YYYY-MM-DD*.json
      4) raw_close_YYYY-MM-DD*.json
    Searches snapshot_dir and snapshot_dir/raw (if present).
    """
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    dirs = _candidate_snapshot_dirs(snapshot_dir)

    for d in pd.date_range(start_date, end_date, freq="D"):
        ymd = d.strftime("%Y%m%d")
        ymd_dash = d.strftime("%Y-%m-%d")

        # CSV close snapshots
        csv_candidates: List[Path] = []
        for base in dirs:
            csv_candidates.append(base / f"close_{ymd}.csv")
        p = _pick_first(csv_candidates)
        if p:
            yield p
            continue

        # JSON close snapshots (close_YYYYMMDD*.json)
        json_candidates: List[Path] = []
        for base in dirs:
            json_candidates.extend(list(base.glob(f"close_{ymd}*.json")))
        p = _pick_first(json_candidates)
        if p:
            yield p
            continue

        # JSON raw snapshots (raw_YYYY-MM-DD*.json)
        raw_candidates: List[Path] = []
        for base in dirs:
            raw_candidates.extend(list(base.glob(f"raw_{ymd_dash}*.json")))
        p = _pick_first(raw_candidates)
        if p:
            yield p
            continue

        # JSON raw_close snapshots (raw_close_YYYY-MM-DD*.json)
        raw_close_candidates: List[Path] = []
        for base in dirs:
            raw_close_candidates.extend(list(base.glob(f"raw_close_{ymd_dash}*.json")))
        p = _pick_first(raw_close_candidates)
        if p:
            yield p
            continue

        # No snapshot for this day → skip


# -----------------------------
# Market feature computation
# -----------------------------

def _market_from_close_csv(path: Path) -> pd.DataFrame:
    """
    Compute consensus + dispersion from a normalized per-book CLOSE CSV snapshot.

    Required columns:
      - merge_key
      - spread_home_point
    """
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    required = {"merge_key", "spread_home_point"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    df = df.copy()
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    df["spread_home_point"] = pd.to_numeric(df["spread_home_point"], errors="coerce")
    df = df.dropna(subset=["merge_key", "spread_home_point"])
    if df.empty:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    out = (
        df.groupby("merge_key")["spread_home_point"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
        .reset_index()
    )
    return out


def _market_from_json(path: Path) -> pd.DataFrame:
    """
    Compute consensus + dispersion from raw JSON using compute_dispersion().
    """
    df = compute_dispersion(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    df = df.rename(
        columns={
            "consensus_close": "home_spread_consensus",
            "book_dispersion": "home_spread_dispersion",
        }
    )
    keep = ["merge_key", "home_spread_consensus", "home_spread_dispersion"]
    df = df[keep].copy()
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    return df


def build_market_df(snapshot_dir: Path, start: str, end: str) -> pd.DataFrame:
    """
    Build a per-merge_key market feature table over the date range.

    Output columns:
      merge_key, home_spread_consensus, home_spread_dispersion
    """
    frames = []
    for path in _iter_snapshot_files(snapshot_dir, start, end):
        try:
            if path.suffix.lower() == ".csv":
                m = _market_from_close_csv(path)
            else:
                m = _market_from_json(path)
            frames.append(m)
        except Exception as e:
            print(f"[build_ats_roi_input] WARNING: failed on {path.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    market_df = pd.concat(frames, ignore_index=True)
    market_df["merge_key"] = market_df["merge_key"].astype(str).str.strip().str.lower()
    market_df = market_df.sort_values("merge_key").drop_duplicates("merge_key", keep="last")
    return market_df


# -----------------------------
# Merge into backtest_joined
# -----------------------------

def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[build_ats_roi_input] Missing date column (expected game_date/date/gamedate).")


def attach_market_features(backtest_path: Path, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach consensus and dispersion to backtest_joined.

    If backtest_joined already has merge_key, we use it.
    Otherwise we compute:
      merge_key = home__away__YYYY-MM-DD (lowercase)
    """
    df = pd.read_csv(backtest_path)
    if df.empty:
        raise RuntimeError("[build_ats_roi_input] backtest_joined is empty")

    if "merge_key" in df.columns:
        df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    else:
        for c in ["home_team", "away_team"]:
            if c not in df.columns:
                raise RuntimeError(f"[build_ats_roi_input] Missing required column: {c}")

        date_col = _detect_date_col(df)
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.isna().any():
            raise RuntimeError("[build_ats_roi_input] Found NaT in date column; cannot build merge_key safely.")
        date_str = dt.dt.strftime("%Y-%m-%d")

        df = df.copy()
        df["merge_key"] = [
            f"{_norm(h)}__{_norm(a)}__{d}"
            for h, a, d in zip(df["home_team"], df["away_team"], date_str)
        ]
        df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()

    merged = df.merge(market_df, on="merge_key", how="left")
    return merged


# -----------------------------
# CLI
# -----------------------------

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)

    consensus_cov = float(merged_df["home_spread_consensus"].notna().mean() * 100.0) if "home_spread_consensus" in merged_df.columns else 0.0
    disp_cov = float(merged_df["home_spread_dispersion"].notna().mean() * 100.0) if "home_spread_dispersion" in merged_df.columns else 0.0
    print(f"[build_ats_roi_input] wrote: {out_path} (consensus_cov={consensus_cov:.1f}% dispersion_cov={disp_cov:.1f}%)")


if __name__ == "__main__":
    main()
