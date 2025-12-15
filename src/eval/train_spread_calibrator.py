"""
Train a bucketed isotonic calibrator mapping ATS residual -> P(home_covers).

Key fix: supports out-of-sample training by date window or by fraction.

Inputs:
- per-game CSV (e.g., outputs/backtest_per_game.csv)

Required columns:
- home_spread_consensus
- fair_spread_model (preferred) OR spread_error (fallback)
- home_score/away_score (or equivalent)
- game_date (preferred). If missing, will error for OOS splits.

ATS outcome:
- home_cover if (home_score + home_spread_consensus) > away_score
- away_cover if (home_score + home_spread_consensus) < away_score
- push if equal (dropped)

Calibrator artifact (joblib dict):
{
  "type": "spread_isotonic_bucketed_v1",
  "global": IsotonicRegression,
  "buckets": [...],
  "meta": {
     "train_start": "...",
     "train_end": "...",
     "n_train": ...,
     ...
  }
}
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


CAL_TYPE = "spread_isotonic_bucketed_v1"
CAL_VERSION = "spread_isotonic_bucketed_v1_2025-12-15_oos_split"


def _find_score_cols(df: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("home_score", "away_score"),
        ("home_pts", "away_pts"),
        ("home_points", "away_points"),
        ("pts_home", "pts_away"),
    ]
    for h, a in candidates:
        if h in df.columns and a in df.columns:
            return h, a
    raise RuntimeError("[train_spread_cal] Missing score columns (home_score/away_score or equivalent).")


def _get_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[train_spread_cal] Missing game_date column required for train/eval splits.")


def _get_residual(df: pd.DataFrame) -> pd.Series:
    if "fair_spread_model" in df.columns and "home_spread_consensus" in df.columns:
        return pd.to_numeric(df["fair_spread_model"], errors="coerce") - pd.to_numeric(df["home_spread_consensus"], errors="coerce")
    if "spread_error" in df.columns:
        return pd.to_numeric(df["spread_error"], errors="coerce")
    raise RuntimeError("[train_spread_cal] Need fair_spread_model+home_spread_consensus OR spread_error.")


def _make_buckets() -> List[Dict[str, Any]]:
    return [
        {"name": "abs0_2p5", "lo": 0.0, "hi": 2.5},
        {"name": "abs2p5_5p5", "lo": 2.5, "hi": 5.5},
        {"name": "abs5p5_8p5", "lo": 5.5, "hi": 8.5},
        {"name": "abs8p5_inf", "lo": 8.5, "hi": 1e9},
    ]


def _bucket_mask(abs_spread: pd.Series, lo: float, hi: float) -> pd.Series:
    return (abs_spread >= lo) & (abs_spread < hi)


def _fit_isotonic(x: np.ndarray, y: np.ndarray, min_samples: int) -> Optional[IsotonicRegression]:
    if len(x) < min_samples:
        return None
    if np.all(y == 0) or np.all(y == 1):
        return None
    x = np.clip(x, -25.0, 25.0)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return model


def _parse_date(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {s}")
    return ts


def main() -> None:
    ap = argparse.ArgumentParser("train_spread_calibrator.py (OOS split capable)")
    ap.add_argument("--per-game", required=True, help="Path to per-game CSV")
    ap.add_argument("--out", default="artifacts/spread_calibrator.joblib", help="Output joblib path")
    ap.add_argument("--min-samples", type=int, default=400, help="Min samples per bucket to fit bucket model")

    # OOS controls
    ap.add_argument("--train-start", default=None, help="Train window start date (YYYY-MM-DD)")
    ap.add_argument("--train-end", default=None, help="Train window end date inclusive (YYYY-MM-DD)")
    ap.add_argument("--train-frac", type=float, default=None, help="Train fraction by date order (e.g., 0.8)")

    args = ap.parse_args()

    if not os.path.exists(args.per_game):
        raise FileNotFoundError(f"[train_spread_cal] per-game not found: {args.per_game}")

    df = pd.read_csv(args.per_game)
    if df.empty:
        raise RuntimeError("[train_spread_cal] per-game CSV is empty.")
    if "home_spread_consensus" not in df.columns:
        raise RuntimeError("[train_spread_cal] Missing required column: home_spread_consensus")

    hcol, acol = _find_score_cols(df)
    date_col = _get_date_col(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise RuntimeError("[train_spread_cal] Found NaT in game_date; cannot do split safely.")

    # Choose training window
    df = df.sort_values(date_col).reset_index(drop=True)

    train_df = df
    train_start = None
    train_end = None

    if args.train_frac is not None:
        if not (0.1 <= float(args.train_frac) <= 0.95):
            raise ValueError("--train-frac should be in [0.1, 0.95]")
        unique_dates = sorted(df[date_col].dt.date.unique())
        cut = int(len(unique_dates) * float(args.train_frac))
        cut = max(1, min(cut, len(unique_dates) - 1))
        end_date = pd.to_datetime(str(unique_dates[cut - 1]))
        train_df = df[df[date_col] <= end_date].copy()
        train_start = str(pd.to_datetime(str(unique_dates[0])).date())
        train_end = str(end_date.date())
    elif args.train_start or args.train_end:
        if not (args.train_start and args.train_end):
            raise ValueError("Provide both --train-start and --train-end")
        ts = _parse_date(args.train_start)
        te = _parse_date(args.train_end)
        train_df = df[(df[date_col] >= ts) & (df[date_col] <= te)].copy()
        train_start = str(ts.date())
        train_end = str(te.date())
    else:
        # Allow legacy behavior but scream loudly
        print("[train_spread_cal] WARNING: No train split provided. This will be IN-SAMPLE if you evaluate on the same file.")
        train_start = str(df[date_col].min().date())
        train_end = str(df[date_col].max().date())

    # Build labels
    home_pts = pd.to_numeric(train_df[hcol], errors="coerce")
    away_pts = pd.to_numeric(train_df[acol], errors="coerce")
    home_spread = pd.to_numeric(train_df["home_spread_consensus"], errors="coerce")
    residual = _get_residual(train_df)

    adj_home = home_pts + home_spread
    is_push = (adj_home == away_pts)
    home_cover = (adj_home > away_pts).astype(int)

    train = pd.DataFrame(
        {
            "residual": residual,
            "home_spread": home_spread,
            "abs_spread": home_spread.abs(),
            "home_cover": home_cover,
            "is_push": is_push,
        }
    )
    train = train.dropna(subset=["residual", "home_spread", "home_cover"])
    train = train[~train["is_push"]].copy()

    if len(train) < 2000:
        print(f"[train_spread_cal] WARNING: training sample seems small (n={len(train)}).")

    x_all = train["residual"].to_numpy(dtype=float)
    y_all = train["home_cover"].to_numpy(dtype=int)

    global_model = _fit_isotonic(x_all, y_all, min_samples=max(int(args.min_samples), 800))
    if global_model is None:
        raise RuntimeError("[train_spread_cal] Failed to fit global isotonic model (insufficient data or degenerate labels).")

    buckets = _make_buckets()
    bucket_models: List[Dict[str, Any]] = []
    for b in buckets:
        m = _bucket_mask(train["abs_spread"], float(b["lo"]), float(b["hi"]))
        bx = train.loc[m, "residual"].to_numpy(dtype=float)
        by = train.loc[m, "home_cover"].to_numpy(dtype=int)
        model = _fit_isotonic(bx, by, min_samples=int(args.min_samples))
        bucket_models.append(
            {"name": b["name"], "lo": float(b["lo"]), "hi": float(b["hi"]), "n": int(len(bx)), "model": model}
        )

    out_obj: Dict[str, Any] = {
        "type": CAL_TYPE,
        "version": CAL_VERSION,
        "global": global_model,
        "buckets": bucket_models,
        "meta": {
            "per_game": args.per_game,
            "date_col": date_col,
            "train_start": train_start,
            "train_end": train_end,
            "n_train": int(len(train)),
            "pushes_dropped": int(is_push.sum()),
            "score_cols": [hcol, acol],
            "uses_spread_error": ("spread_error" in train_df.columns and "fair_spread_model" not in train_df.columns),
            "notes": "Maps residual=fair_spread_model-home_spread_consensus to P(home_covers).",
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(out_obj, args.out)
    print(f"[train_spread_cal] wrote: {args.out}")
    print(f"[train_spread_cal] train_range={train_start}..{train_end} n_train={len(train)}")
    print(f"[train_spread_cal] buckets_fit={[{'name':b['name'],'n':b['n'],'fit':b['model'] is not None} for b in bucket_models]}")
    print("[train_spread_cal] DONE")


if __name__ == "__main__":
    main()
