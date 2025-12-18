#!/usr/bin/env python3
"""
Historical prediction runner.

Goal (Commit 2): given a historical games CSV, generate per-day prediction CSVs in outputs/
using the existing `predict_games` function, and (optionally) apply market snapshots.

This file is intentionally defensive about imports + column names so it runs in Colab/GitHub
without requiring the repo to be installed as a package.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------
# Import predict_games safely
# ---------------------------
def _import_predict_games():
    """
    Supports both layouts:
      - repo_root/predict.py
      - repo_root/src/predict.py  (package-style)
    """
    try:
        from src.predict import predict_games  # type: ignore
        return predict_games
    except Exception:
        pass
    try:
        from predict import predict_games  # type: ignore
        return predict_games
    except Exception as e:
        raise ModuleNotFoundError(
            "Could not import predict_games. Expected either src/predict.py (import src.predict) "
            "or predict.py at repo root (import predict)."
        ) from e


predict_games = _import_predict_games()


# ---------------------------
# Utilities
# ---------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_date_col(df: pd.DataFrame) -> str:
    for c in ("date", "game_date", "gameDate", "start_date"):
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a date column. Available columns: {list(df.columns)[:40]}...")


def _normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out = out.dropna(subset=[date_col])
    return out


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}. "
                         f"Have: {list(df.columns)[:60]}...")


# ---------------------------
# Market snapshot application
# ---------------------------
def _load_close_snapshot(snapshot_dir: Path, day) -> Optional[pd.DataFrame]:
    """
    Expects files like close_YYYYMMDD.csv in snapshot_dir.
    Returns a normalized DF with date, home_team, away_team, market_* columns if found.
    """
    ymd = pd.Timestamp(day).strftime("%Y%m%d")
    p = snapshot_dir / f"close_{ymd}.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)

    # Try to standardize to home_team/away_team + market_spread/market_total (+ optional moneyline)
    # We don't assume exact schema; we map a few common ones.
    col_map = {}
    # team columns
    for home_c in ("home_team", "home", "homeTeam", "team_home", "home_abbr"):
        if home_c in df.columns:
            col_map[home_c] = "home_team"
            break
    for away_c in ("away_team", "away", "awayTeam", "team_away", "away_abbr", "visitor_abbr"):
        if away_c in df.columns:
            col_map[away_c] = "away_team"
            break

    # spread / total columns (close lines)
    for s_c in ("close_spread", "spread_close", "spread", "closing_spread", "home_spread"):
        if s_c in df.columns:
            col_map[s_c] = "market_spread"
            break
    for t_c in ("close_total", "total_close", "total", "closing_total"):
        if t_c in df.columns:
            col_map[t_c] = "market_total"
            break

    # moneyline columns are messy; we keep them only if present
    for mlh in ("home_ml", "home_moneyline", "moneyline_home", "close_home_ml"):
        if mlh in df.columns:
            col_map[mlh] = "market_home_ml"
            break
    for mla in ("away_ml", "away_moneyline", "moneyline_away", "close_away_ml"):
        if mla in df.columns:
            col_map[mla] = "market_away_ml"
            break

    if "home_team" not in col_map.values() or "away_team" not in col_map.values():
        # Can't merge safely; skip.
        return None

    df2 = df.rename(columns=col_map).copy()
    df2["date"] = pd.Timestamp(day).date()

    keep = ["date", "home_team", "away_team"] + [c for c in ("market_spread", "market_total", "market_home_ml", "market_away_ml") if c in df2.columns]
    return df2[keep]


def _apply_market(df_pred: pd.DataFrame, snapshot_dir: Path, day) -> pd.DataFrame:
    snap = _load_close_snapshot(snapshot_dir, day)
    if snap is None:
        print(f"[historical][WARNING] No CLOSE odds snapshot found for {day} in {snapshot_dir} — skipping market ensemble")
        return df_pred

    # Merge on (date, home_team, away_team)
    merged = df_pred.merge(
        snap,
        how="left",
        on=["date", "home_team", "away_team"],
        suffixes=("", "_m"),
    )
    return merged


# ---------------------------
# Runner
# ---------------------------
def _run_day(df_day: pd.DataFrame, apply_market: bool, snapshot_dir: Path) -> pd.DataFrame:
    # predict_games expects home_team + away_team (per src/predict.py)
    _require_cols(df_day, ["home_team", "away_team"], "history slice")

    df_in = df_day[["home_team", "away_team"]].copy()
    df_out = predict_games(df_in)  # NOTE: DO NOT pass extra kwargs; signature is positional-only.

    # Ensure date column exists for downstream join/backtest
    df_out = df_out.copy()
    df_out["date"] = df_day["date"].iloc[0]

    if apply_market:
        df_out = _apply_market(df_out, snapshot_dir=snapshot_dir, day=df_day["date"].iloc[0])

    return df_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to games_2019_2024.csv (or similar)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out-dir", default="outputs", help="Where to write predictions_YYYY-MM-DD.csv files")
    ap.add_argument("--apply-market", action="store_true", help="Merge in close_YYYYMMDD.csv snapshots from --snapshot-dir")
    ap.add_argument("--snapshot-dir", default="data/_snapshots", help="Directory containing close_YYYYMMDD.csv files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing daily prediction files")
    ap.add_argument("--audit-path", default="outputs/historical_prediction_runner_audit.json", help="Audit JSON output")
    args = ap.parse_args()

    hist_path = Path(args.history)
    out_dir = Path(args.out_dir)
    snapshot_dir = Path(args.snapshot_dir)
    audit_path = Path(args.audit_path)

    _ensure_dir(out_dir)
    _ensure_dir(audit_path.parent)

    df = pd.read_csv(hist_path)
    date_col = _parse_date_col(df)
    df = _normalize_dates(df, date_col=date_col).rename(columns={date_col: "date"})

    # Basic normalization: ensure team columns exist
    if "home_team" not in df.columns or "away_team" not in df.columns:
        # try common alternatives
        ren = {}
        for hc in ("home_team", "homeTeam", "home", "team_home"):
            if hc in df.columns:
                ren[hc] = "home_team"
                break
        for ac in ("away_team", "awayTeam", "away", "team_away", "visitor_team"):
            if ac in df.columns:
                ren[ac] = "away_team"
                break
        df = df.rename(columns=ren)

    _require_cols(df, ["date", "home_team", "away_team"], "history")

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    if df.empty:
        raise SystemExit(f"No rows in history after filtering to {start}..{end}")

    # Run per day
    days = sorted(df["date"].unique())
    audit: Dict[str, object] = {
        "history_path": str(hist_path),
        "start": str(start),
        "end": str(end),
        "out_dir": str(out_dir),
        "snapshot_dir": str(snapshot_dir),
        "apply_market": bool(args.apply_market),
        "days": [],
    }

    print("[historical] loaded models via predict_games()")
    for day in days:
        daily_path = out_dir / f"predictions_{day}.csv"
        if daily_path.exists() and not args.overwrite:
            print(f"[historical] exists {daily_path} (skip; use --overwrite)")
            continue

        df_day = df[df["date"] == day].copy()
        df_pred = _run_day(df_day, apply_market=bool(args.apply_market), snapshot_dir=snapshot_dir)

        df_pred.to_csv(daily_path, index=False)
        print(f"[historical] Wrote {daily_path} ({len(df_pred)} rows)")
        audit["days"].append({"date": str(day), "rows": int(len(df_pred)), "path": str(daily_path)})

    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[historical] wrote {audit_path}")


if __name__ == "__main__":
    main()
