"""
Train a bucketed isotonic calibrator mapping residual -> P(home_covers).

Inputs:
- outputs/backtest_per_game.csv (or any per-game file with required columns)

Required columns:
- home_spread_consensus
- fair_spread_model  (or spread_error as fallback)
- home_score, away_score  (or home_pts/away_pts if you use those names)

We compute:
- residual = fair_spread_model - home_spread_consensus
  (if spread_error exists and fair_spread_model is missing, we use spread_error)

ATS outcome:
- home_cover if (home_score + home_spread_consensus) > away_score
- away_cover if (home_score + home_spread_consensus) < away_score
- push if equal (dropped from training)

We fit isotonic regression per bucket of abs(home_spread_consensus).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


CAL_VERSION = "spread_isotonic_bucketed_v1_2025-12-15"


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


def _get_residual(df: pd.DataFrame) -> pd.Series:
    if "fair_spread_model" in df.columns and "home_spread_consensus" in df.columns:
        return pd.to_numeric(df["fair_spread_model"], errors="coerce") - pd.to_numeric(df["home_spread_consensus"], errors="coerce")
    if "spread_error" in df.columns:
        return pd.to_numeric(df["spread_error"], errors="coerce")
    raise RuntimeError("[train_spread_cal] Need fair_spread_model+home_spread_consensus OR spread_error.")


def _make_buckets() -> List[Dict[str, Any]]:
    # abs(spread) buckets — tuned for NBA typical distribution
    return [
        {"name": "abs0_2p5", "lo": 0.0, "hi": 2.5},
        {"name": "abs2p5_5p5", "lo": 2.5, "hi": 5.5},
        {"name": "abs5p5_8p5", "lo": 5.5, "hi": 8.5},
        {"name": "abs8p5_inf", "lo": 8.5, "hi": 1e9},
    ]


def _bucket_mask(abs_spread: pd.Series, lo: float, hi: float) -> pd.Series:
    return (abs_spread >= lo) & (abs_spread < hi)


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> Optional[IsotonicRegression]:
    # isotonic needs at least some positive/negative and enough data
    if len(x) < 200:
        return None
    if np.all(y == 0) or np.all(y == 1):
        return None
    # clip x to reduce extreme leverage
    x = np.clip(x, -25.0, 25.0)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return model


def main() -> None:
    ap = argparse.ArgumentParser("train_spread_calibrator.py")
    ap.add_argument("--per-game", required=True, help="Path to per-game CSV (e.g., outputs/backtest_per_game.csv)")
    ap.add_argument("--out", default="artifacts/spread_calibrator.joblib", help="Output path for calibrator joblib")
    ap.add_argument("--min-samples", type=int, default=200, help="Min samples per bucket to fit a bucket model")
    args = ap.parse_args()

    if not os.path.exists(args.per_game):
        raise FileNotFoundError(f"[train_spread_cal] per-game not found: {args.per_game}")

    df = pd.read_csv(args.per_game)
    if df.empty:
        raise RuntimeError("[train_spread_cal] per-game CSV is empty.")

    if "home_spread_consensus" not in df.columns:
        raise RuntimeError("[train_spread_cal] Missing required column: home_spread_consensus")

    hcol, acol = _find_score_cols(df)
    home_pts = pd.to_numeric(df[hcol], errors="coerce")
    away_pts = pd.to_numeric(df[acol], errors="coerce")
    home_spread = pd.to_numeric(df["home_spread_consensus"], errors="coerce")

    residual = _get_residual(df)

    # ATS outcome
    adj_home = home_pts + home_spread
    is_push = (adj_home == away_pts)
    home_cover = (adj_home > away_pts).astype(int)

    # Training frame
    train = pd.DataFrame(
        {
            "residual": residual,
            "home_spread": home_spread,
            "abs_spread": home_spread.abs(),
            "home_cover": home_cover,
            "is_push": is_push,
        }
    )

    # drop missing & pushes
    train = train.dropna(subset=["residual", "home_spread", "home_cover"])
    train = train[~train["is_push"]].copy()

    if len(train) < 1000:
        print(f"[train_spread_cal] WARNING: training sample seems small (n={len(train)}).")

    x_all = train["residual"].to_numpy(dtype=float)
    y_all = train["home_cover"].to_numpy(dtype=int)

    # Global model
    global_model = _fit_isotonic(x_all, y_all)
    if global_model is None:
        raise RuntimeError("[train_spread_cal] Failed to fit global isotonic model (insufficient data or degenerate labels).")

    buckets = _make_buckets()
    bucket_models: List[Dict[str, Any]] = []
    for b in buckets:
        m = _bucket_mask(train["abs_spread"], float(b["lo"]), float(b["hi"]))
        bx = train.loc[m, "residual"].to_numpy(dtype=float)
        by = train.loc[m, "home_cover"].to_numpy(dtype=int)

        # enforce min samples
        if len(bx) < int(args.min_samples):
            model = None
        else:
            model = _fit_isotonic(bx, by)

        bucket_models.append(
            {
                "name": b["name"],
                "lo": float(b["lo"]),
                "hi": float(b["hi"]),
                "n": int(len(bx)),
                "model": model,
            }
        )

    out_obj: Dict[str, Any] = {
        "type": "spread_isotonic_bucketed_v1",
        "version": CAL_VERSION,
        "global": global_model,
        "buckets": bucket_models,
        "meta": {
            "per_game": args.per_game,
            "score_cols": [hcol, acol],
            "uses_spread_error": ("spread_error" in df.columns and "fair_spread_model" not in df.columns),
            "n_train": int(len(train)),
            "pushes_dropped": int(is_push.sum()),
            "notes": "Maps residual=fair_spread_model-home_spread_consensus to P(home_covers).",
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(out_obj, args.out)
    print(f"[train_spread_cal] wrote: {args.out}")
    print(f"[train_spread_cal] n_train={len(train)} global_fit=OK buckets_fit={[{'name':b['name'],'n':b['n'],'fit':b['model'] is not None} for b in bucket_models]}")
    print("[train_spread_cal] DONE")


if __name__ == "__main__":
    main()
