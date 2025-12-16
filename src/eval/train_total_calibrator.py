"""
Train isotonic calibrator for NBA totals (Over/Under).

Maps:
    total_residual -> P(over)

Leak-safe:
- Explicit train_start / train_end
- No eval contamination
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def main() -> None:
    ap = argparse.ArgumentParser("train_total_calibrator.py")
    ap.add_argument("--per-game", required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    df = pd.read_csv(args.per_game)
    if df.empty:
        raise RuntimeError("[total_cal] per_game is empty")

    # date handling
    date_col = "game_date" if "game_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    ts = pd.to_datetime(args.train_start)
    te = pd.to_datetime(args.train_end)

    train = df[(df[date_col] >= ts) & (df[date_col] <= te)].copy()
    if train.empty:
        raise RuntimeError("[total_cal] no rows in training window")

    # required columns
    required = [
        "fair_total_model",
        "total_consensus",
        "home_score",
        "away_score",
    ]
    for c in required:
        if c not in train.columns:
            raise RuntimeError(f"[total_cal] missing required column: {c}")

    train["actual_total"] = train["home_score"] + train["away_score"]
    train["total_residual"] = train["fair_total_model"] - train["total_consensus"]
    train["went_over"] = (train["actual_total"] > train["total_consensus"]).astype(int)

    X = train["total_residual"].to_numpy(dtype=float)
    y = train["went_over"].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise RuntimeError("[total_cal] degenerate labels (all overs or unders)")

    if len(X) < 150:
        print(f"[total_cal] WARNING: small sample size (n={len(X)})")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)

    artifact: Dict = {
        "type": "total_isotonic",
        "train_start": str(ts.date()),
        "train_end": str(te.date()),
        "n_samples": int(len(X)),
        "model": iso,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(artifact, args.out)

    print(f"[total_cal] trained totals calibrator")
    print(f"[total_cal] rows={len(X)} window={ts.date()}..{te.date()}")
    print(f"[total_cal] wrote: {args.out}")


if __name__ == "__main__":
    main()

