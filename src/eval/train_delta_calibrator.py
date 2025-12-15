"""
Train a market-relative calibrator for NBA Pro‑Lite/Elite models.

This script reads a per-game backtest CSV (e.g. ``outputs/backtest_per_game.csv``),
computes the delta between the model’s raw probability and the de‑vigged market
probability, buckets games by moneyline odds magnitude, and fits isotonic
regression calibrators per bucket.  The resulting calibrator dictionary is
saved to disk via ``joblib``.

Usage (from repository root):

    PYTHONPATH=. python -m src.eval.train_delta_calibrator \
        --per_game outputs/backtest_per_game.csv \
        --out artifacts/delta_calibrator.joblib

You can then pass the saved ``delta_calibrator.joblib`` to the ROI analysis
module when running in ``ev_cal`` mode to produce calibrated expected-value
bet selection.
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


def _infer_model_prob_column(df: pd.DataFrame) -> str:
    """Infer the model probability column to use.

    Preference order:
        1. ``home_win_prob_model_raw``
        2. ``home_win_prob_model``
        3. ``home_win_prob``

    Raises if none are found.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser("train_delta_calibrator.py")
    ap.add_argument(
        "--per_game",
        required=True,
        help="Path to per-game backtest CSV (must include model and market probabilities and scores).",
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
    args = ap.parse_args()

    per_game_path: str = args.per_game
    out_path: str = args.out
    min_samples: int = int(args.min_samples)

    if not os.path.exists(per_game_path):
        raise FileNotFoundError(f"[train_delta_calibrator] per_game file not found: {per_game_path}")

    df = pd.read_csv(per_game_path)

    # Ensure required columns exist
    # Determine model probability column
    model_prob_col = _infer_model_prob_column(df)
    # Copy to expected column names
    df = df.copy()
    df["model_prob_home_raw"] = pd.to_numeric(df[model_prob_col], errors="coerce")
    # Use market probability column if present (home_ml_prob_consensus) else compute from ML odds
    if "market_prob_home" not in df.columns:
        # Attempt to compute from ml_home_consensus and ml_away_consensus
        if "ml_home_consensus" in df.columns and "ml_away_consensus" in df.columns:
            from src.eval.roi_analysis import devig_home_prob  # type: ignore

            df["market_prob_home"], _method = zip(
                *df.apply(
                    lambda r: devig_home_prob(r.get("ml_home_consensus"), r.get("ml_away_consensus")),
                    axis=1,
                )
            )
        else:
            raise RuntimeError(
                "[train_delta_calibrator] market_prob_home column missing and cannot compute from ML odds"
            )

    # Ensure home_win_actual exists; compute if needed
    if "home_win_actual" not in df.columns:
        if "home_score" in df.columns and "away_score" in df.columns:
            df["home_win_actual"] = (
                pd.to_numeric(df["home_score"], errors="coerce")
                > pd.to_numeric(df["away_score"], errors="coerce")
            ).astype(float)
        else:
            raise RuntimeError(
                "[train_delta_calibrator] home_win_actual missing and cannot infer from scores"
            )

    # Fit calibrator
    calibrator = fit_delta_calibrator(df, min_samples=min_samples)
    save_delta_calibrator(calibrator, out_path)
    print(
        f"[train_delta_calibrator] Saved delta calibrator with {len(calibrator.get('calibrators', {}))} buckets "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
