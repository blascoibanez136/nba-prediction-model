"""Run day-by-day historical predictions over a games history CSV.

Commit-2 goal:
- Load models from local artifact dirs (models/ or artifacts/models/).
- For each day in [start, end], run predict.predict_games(df_day) to produce model outputs.
- Optionally apply market snapshot (close_YYYYMMDD.csv) when available.
- Write per-day predictions CSVs to outputs/ and emit an audit JSON.

This module intentionally DOES NOT require predict.predict_games() to accept models/teams kwargs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.predict import predict_games


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = REPO_DIR / "data" / "history" / "games_2019_2024.csv"
DEFAULT_OUT_DIR = REPO_DIR / "outputs"
DEFAULT_SNAP_DIR = REPO_DIR / "data" / "_snapshots"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_history_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif "game_date" in df.columns:
        df["date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df.rename(columns={"game_date": "date"})
    else:
        raise ValueError(f"History CSV missing a date column. Columns={list(df.columns)}")
    return df


def _load_market_snapshot(snap_dir: Path, d: date) -> Optional[pd.DataFrame]:
    # We expect close_YYYYMMDD.csv
    fn = f"close_{d.strftime('%Y%m%d')}.csv"
    p = snap_dir / fn
    if not p.exists():
        return None
    try:
        mkt = pd.read_csv(p)
    except Exception:
        return None
    return mkt


def _apply_market_ensemble(df_pred: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """Lightweight, safe market merge.

    We *do not* assume a particular schema beyond having a join key. The code tries
    common keys and only adds columns when merge succeeds.
    """
    if mkt is None or len(mkt) == 0:
        return df_pred

    # Try join keys in order
    join_pairs = [
        ("game_id", "game_id"),
        ("id", "id"),
        ("game_id", "id"),
        ("id", "game_id"),
    ]
    left = df_pred.copy()
    merged = None
    for lk, rk in join_pairs:
        if lk in left.columns and rk in mkt.columns:
            try:
                merged = left.merge(mkt, how="left", left_on=lk, right_on=rk, suffixes=("", "_mkt"))
                break
            except Exception:
                merged = None

    if merged is None:
        return df_pred

    # No-op: we don't compute edges here (that's backtest/analysis territory).
    # But we keep market columns so downstream can use them.
    return merged


def _run_day(df_day: pd.DataFrame, apply_market: bool, snap_dir: Path, d: date) -> pd.DataFrame:
    out = predict_games(df_day)

    # predict_games may return dict-like; normalize to DataFrame
    if isinstance(out, dict):
        out_df = pd.DataFrame([out])
    else:
        out_df = out

    if apply_market:
        mkt = _load_market_snapshot(snap_dir, d)
        if mkt is None:
            print(f"[historical][WARNING] No CLOSE odds snapshot found for {_ymd(d)} in {snap_dir} – skipping market ensemble")
        else:
            out_df = _apply_market_ensemble(out_df, mkt)

    return out_df


def _date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=str, default=str(DEFAULT_HISTORY))
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--snapshots", type=str, default=str(DEFAULT_SNAP_DIR))
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    history_path = Path(args.history)
    out_dir = Path(args.out_dir)
    snap_dir = Path(args.snapshots)
    _ensure_dir(out_dir)

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    df = _read_history_csv(history_path)
    df = _date_range(df, start, end)

    # Require at least one of these ids for stable file writing & joins
    if "game_id" not in df.columns and "id" not in df.columns:
        raise ValueError("History CSV must include game_id or id for joining.")

    # Run day-by-day to keep files small and make debugging easier
    audit: Dict[str, Any] = {
        "history": str(history_path),
        "out_dir": str(out_dir),
        "snapshots": str(snap_dir),
        "start": _ymd(start),
        "end": _ymd(end),
        "apply_market": bool(args.apply_market),
        "days": [],
    }

    for d, df_day in df.groupby("date"):
        d = d if isinstance(d, date) else pd.to_datetime(d).date()
        out_path = out_dir / f"predictions_{_ymd(d)}.csv"
        if out_path.exists() and not args.overwrite:
            print(f"[historical] exists {out_path} (skipping; use --overwrite)")
            audit["days"].append({"date": _ymd(d), "skipped": True, "out": str(out_path)})
            continue

        df_pred = _run_day(df_day, args.apply_market, snap_dir, d)
        df_pred.to_csv(out_path, index=False)
        print(f"[historical] Wrote {out_path} ({len(df_pred)} rows)")

        audit["days"].append({"date": _ymd(d), "skipped": False, "out": str(out_path), "rows": int(len(df_pred))})

    audit_path = out_dir / "historical_prediction_runner_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[historical] wrote {audit_path}")


if __name__ == "__main__":
    main()
