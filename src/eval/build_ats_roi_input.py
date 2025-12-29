"""src/eval/build_ats_roi_input.py

Merge backtest data with market consensus and dispersion for ATS ROI.

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

import pandas as pd

from src.ingest.odds_snapshots import compute_dispersion, _norm

def _iter_snapshot_files(snapshot_dir: Path, start: str, end: str):
    # Yield the first close snapshot for each date in range.
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    for date in pd.date_range(start_date, end_date, freq="D"):
        dstr = date.strftime("%Y%m%d")
        files = sorted(p for p in snapshot_dir.glob(f"close_{dstr}*.json") if p.is_file())
        if files:
            yield files[0]

def build_market_df(snapshot_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for path in _iter_snapshot_files(snapshot_dir, start, end):
        df = compute_dispersion(path).rename(
            columns={"consensus_close": "home_spread_consensus", "book_dispersion": "home_spread_dispersion"}
        )
        frames.append(df[["merge_key", "home_spread_consensus", "home_spread_dispersion"]])
    if not frames:
        return pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])
    market_df = pd.concat(frames, ignore_index=True)
    # keep the last observation per merge_key
    return market_df.sort_values("merge_key").drop_duplicates("merge_key", keep="last")

def attach_market_features(backtest_path: Path, market_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(backtest_path)
    def mk(row):
        return f"{_norm(row['home_team'])}__{_norm(row['away_team'])}__{row['game_date']}"
    df["merge_key"] = df.apply(mk, axis=1)
    return df.merge(market_df, on="merge_key", how="left")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest-joined", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    market_df = build_market_df(Path(args.snapshot_dir), args.start, args.end)
    merged_df = attach_market_features(Path(args.backtest_joined), market_df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(args.out, index=False)

    cov_consensus = merged_df["home_spread_consensus"].notna().mean() * 100
    cov_disp = merged_df["home_spread_dispersion"].notna().mean() * 100
    print(f"[build_ats_roi_input] wrote: {args.out} (consensus_cov={cov_consensus:.1f}% dispersion_cov={cov_disp:.1f}%)")

if __name__ == "__main__":
    main()
