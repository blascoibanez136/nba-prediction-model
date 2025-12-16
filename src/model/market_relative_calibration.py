"""
Market-relative calibration utilities for NBA Pro-Lite/Elite models.

Fix (CRITICAL):
- Training and inference must use the SAME delta transform.
- We train isotonic on x = (clip(delta,[-1,1]) + 1) / 2 in [0,1],
  and apply the same transform at inference.

This prevents a scale mismatch where training used raw delta but inference
used mapped delta, which silently breaks calibration.

The calibrator is stored as:
{
  "calibrators": {bucket_name: IsotonicRegression, ...},
  "min_samples": int,
  "meta": {...}
}
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.model.calibration import fit_isotonic, apply_calibrator

PROB_EPS = 1e-6


###############################################################################
# Bucketing logic
###############################################################################
def _bucket_from_odds(odds: Optional[float]) -> Optional[str]:
    """Compute a bucket label from American odds.

    Buckets:
        fav_small : -100 >= odds > -200
        fav_medium: -200 >= odds > -400
        fav_big   : odds <= -400
        dog_small : 100 <= odds < 200
        dog_medium: 200 <= odds < 400
        dog_big   : odds >= 400
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if np.isnan(o) or np.isinf(o) or abs(o) < 100:
        return None

    if o < 0:
        mag = abs(o)
        if mag <= 200:
            return "fav_small"
        if mag <= 400:
            return "fav_medium"
        return "fav_big"
    else:
        mag = o
        if mag < 200:
            return "dog_small"
        if mag < 400:
            return "dog_medium"
        return "dog_big"


###############################################################################
# Shared transforms (MUST match training and inference)
###############################################################################
def _clip_prob(v: float) -> float:
    return float(np.clip(v, PROB_EPS, 1.0 - PROB_EPS))


def _delta_to_x(delta: float) -> float:
    """
    Map delta in [-1,1] to x in [0,1] with clipping.
    This is the ONLY representation we train/apply isotonic on.
    """
    d = float(np.clip(delta, -1.0, 1.0))
    return (d + 1.0) / 2.0


###############################################################################
# Training and saving calibrators
###############################################################################
def fit_delta_calibrator(df: pd.DataFrame, *, min_samples: int = 25) -> Dict[str, object]:
    """Fit per-bucket isotonic calibrators on model vs market probability delta.

    Required columns:
      model_prob_home_raw, market_prob_home, home_win_actual, ml_home_consensus

    IMPORTANT:
      We fit isotonic on x = (clip(delta)+1)/2 in [0,1],
      not on raw delta, so inference can use the same mapping.
    """
    required_cols = {"model_prob_home_raw", "market_prob_home", "home_win_actual", "ml_home_consensus"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"[delta_calibrator] Missing required columns: {missing}")

    work = df.copy()

    work["model_prob_home_raw"] = pd.to_numeric(work["model_prob_home_raw"], errors="coerce")
    work["market_prob_home"] = pd.to_numeric(work["market_prob_home"], errors="coerce")
    work["home_win_actual"] = pd.to_numeric(work["home_win_actual"], errors="coerce")
    work["ml_home_consensus"] = pd.to_numeric(work["ml_home_consensus"], errors="coerce")

    # clip probabilities to sane bounds before delta
    work = work.dropna(subset=["model_prob_home_raw", "market_prob_home", "home_win_actual", "ml_home_consensus"]).copy()
    if work.empty:
        raise RuntimeError("[delta_calibrator] No valid rows after dropping NaNs")

    work["model_prob_home_raw"] = work["model_prob_home_raw"].clip(PROB_EPS, 1.0 - PROB_EPS)
    work["market_prob_home"] = work["market_prob_home"].clip(PROB_EPS, 1.0 - PROB_EPS)

    work["delta"] = work["model_prob_home_raw"] - work["market_prob_home"]
    work["x"] = work["delta"].apply(_delta_to_x)
    work["bucket"] = work["ml_home_consensus"].apply(_bucket_from_odds)

    calibrators: Dict[str, object] = {}
    bucket_counts: Dict[str, int] = {}

    for bucket, group in work.groupby("bucket"):
        if bucket is None:
            continue

        sub = group.loc[:, ["x", "home_win_actual"]].dropna()
        n = int(len(sub))
        bucket_counts[str(bucket)] = n

        if n < min_samples:
            continue

        # Fit isotonic regression calibrator on x -> win probability
        calib = fit_isotonic(sub["home_win_actual"].values, sub["x"].values)
        calibrators[str(bucket)] = calib

    return {
        "calibrators": calibrators,
        "min_samples": int(min_samples),
        "meta": {
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "transform": "x=(clip(delta,[-1,1])+1)/2",
            "prob_clip_eps": PROB_EPS,
            "bucket_counts": bucket_counts,
            "buckets_fitted": sorted(list(calibrators.keys())),
        },
    }


def save_delta_calibrator(calibrator: Dict[str, object], path: str) -> None:
    """Serialize a delta calibrator dictionary to disk via joblib."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(calibrator, path)


def load_delta_calibrator(path: str) -> Dict[str, object]:
    """Load a delta calibrator dictionary from disk."""
    return joblib.load(path)


###############################################################################
# Applying calibrators
###############################################################################
def apply_delta_calibrator(
    delta: Optional[float],
    ml_home_odds: Optional[float],
    calibrator: Dict[str, object],
) -> Optional[float]:
    """Apply a delta calibrator to produce a calibrated home win probability.

    Returns:
        Calibrated home win probability in [0,1], or None if unavailable.
    """
    if delta is None or ml_home_odds is None:
        return None
    try:
        d = float(delta)
        o = float(ml_home_odds)
    except Exception:
        return None
    if np.isnan(d) or np.isinf(d):
        return None

    bucket = _bucket_from_odds(o)
    calibrators = calibrator.get("calibrators", {})
    if bucket is None or bucket not in calibrators:
        return None

    iso = calibrators[bucket]

    # Apply the SAME transform used in training
    x = _delta_to_x(d)
    p_cal = apply_calibrator(np.array([x], dtype=float), iso)[0]

    # Ensure within [0,1]
    return float(np.clip(p_cal, 0.0, 1.0))
