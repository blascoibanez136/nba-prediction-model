"""
Train a market-relative calibrator for NBA Pro-Lite/Elite models.

Reads per-game backtest CSV, computes:
  delta = p_model_raw - p_market_devig

Buckets by ML odds magnitude and fits isotonic calibrators per bucket.
Saves a dict artifact via joblib.

Hardening:
- Optional train window filtering (--train-start/--train-end) with NaT drop
- Uses shared odds devig helper (src.utils.odds_math) to avoid circular imports
- Clips model probs to (1e-6, 1-1e-6) before fitting

Usage:
    PYTHONPATH=. python -m src.eval.train_delta_calibrator \
        --per_game outputs/backtest_per_game.csv \
        --out artifacts/delta_calibrator.joblib \
        --train-start 2023-10-24 --train-end 2024-03-10
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

from src.model.market_relative_calibration import (
    fit_delta_calibrator,
    save_delta_calibrator,
)
from src.utils.odds_math import devig_home_prob

PROB_EPS = 1e-6


def _infer_model_prob_column(df: pd.DataFrame) -> str:
    """Infer the model probability column to use."""
    candidates = [
        "home_win_prob_model_raw",
        "home_win_prob_model",
        "home_win_prob",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"[train_delta_calibrator] No model probability column found. Tried: {candidates}"
    )


def _get_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    return None


def _clip_prob(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(PROB_EPS, 1.0 - PROB_EPS)


def main() -> None:
    ap = argparse.ArgumentParser("train_delta_calibrator.py")
    ap.add_argument(
        "--per_game",
        required=True,
        help="Path to per-game backtest CSV (must include model probs, ML odds, and scores).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output path for the saved calibrator (e.g. artifacts/delta_calibrator.joblib).",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=25,
        help="Minimum number of samples per bucket to fit a calibrator. Buckets with fewer samples are skipped.",
    )

    # Hardening: train window filtering (optional)
    ap.add_argument("--train-start", default=None, help="Filter training start date (YYYY-MM-DD). Optional.")
    ap.add_argument("--train-end", default=None, help="Filter training end date (YYYY-MM-DD). Optional.")

    args = ap.parse_args()

    per_game_path: str = args.per_game
    out_path: str = args.out
    min_samples: int = int(args.min_samples)

    if not os.path.exists(per_game_path):
        raise FileNotFoundError(f"[train_delta_calibrator] per_game file not found: {per_game_path}")

    df = pd.read_csv(per_game_path)
    if df.empty:
        raise RuntimeError("[train_delta_calibrator] per_game is empty")

    # Optional train window filter
    date_col = _get_date_col(df)
    if (args.train_start or args.train_end) and not date_col:
        raise RuntimeError("[train_delta_calibrator] train-start/train-end provided but no date column found")

    if date_col and (args.train_start or args.train_end):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()

        ts = pd.to_datetime(args.train_start, errors="coerce") if args.train_start else None
        te = pd.to_datetime(args.train_end, errors="coerce") if args.train_end else None
        if args.train_start and pd.isna(ts):
            raise RuntimeError("[train_delta_calibrator] invalid train-start")
        if args.train_end and pd.isna(te):
            raise RuntimeError("[train_delta_calibrator] invalid train-end")

        if ts is not None:
            df = df[df[date_col] >= ts].copy()
        if te is not None:
            df = df[df[date_col] <= te].copy()

        print(
            f"[train_delta_calibrator] train_window: start={str(ts.date()) if ts is not None else None} "
            f"end={str(te.date()) if te is not None else None} rows={len(df)}"
        )

    if df.empty:
        raise RuntimeError("[train_delta_calibrator] No rows after train window filtering")

    # Determine model probability column and normalize to expected name
    model_prob_col = _infer_model_prob_column(df)
    df = df.copy()
    df["model_prob_home_raw"] = _clip_prob(df[model_prob_col])

    # Ensure market_prob_home exists or compute from ML odds
    if "market_prob_home" not in df.columns:
        if "ml_home_consensus" in df.columns and "ml_away_consensus" in df.columns:
            dev = df.apply(
                lambda r: devig_home_prob(r.get("ml_home_consensus"), r.get("ml_away_consensus")),
                axis=1,
                result_type="expand",
            )
            df["market_prob_home"] = dev[0]
        else:
            raise RuntimeError(
                "[train_delta_calibrator] market_prob_home missing and cannot compute (ml_home_consensus/ml_away_consensus missing)"
            )

    df["market_prob_home"] = _clip_prob(df["market_prob_home"])

    # Ensure home_win_actual exists; compute if needed
    if "home_win_actual" not in df.columns:
        if "home_score" in df.columns and "away_score" in df.columns:
            df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
            df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
            df = df.dropna(subset=["home_score", "away_score"]).copy()
            df["home_win_actual"] = (df["home_score"] > df["away_score"]).astype(float)
        else:
            raise RuntimeError("[train_delta_calibrator] home_win_actual missing and cannot infer from scores")

    if df.empty:
        raise RuntimeError("[train_delta_calibrator] No valid rows after score coercion")

    # Fit calibrator
    calibrator = fit_delta_calibrator(df, min_samples=min_samples)
    save_delta_calibrator(calibrator, out_path)

    n_buckets = len(calibrator.get("calibrators", {}))
    print(f"[train_delta_calibrator] Saved delta calibrator with {n_buckets} buckets to {out_path}")


if __name__ == "__main__":
    main()
