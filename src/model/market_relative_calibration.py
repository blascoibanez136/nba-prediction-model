"""
Market-relative calibration utilities for NBA Pro‑Lite/Elite models.

This module provides helpers to train and apply a calibration model on the
delta between the model’s raw win probability and the de‑vigged market
probability.  The intent is to anchor a model’s probability to market
efficiency rather than calibrating absolute probabilities in isolation.

The primary functions are:

* ``fit_delta_calibrator`` – Fit isotonic regression calibrators on
  ``delta = p_model - p_market`` separately for buckets of moneyline odds.
* ``save_delta_calibrator`` / ``load_delta_calibrator`` – Serialize and
  deserialize the calibrator to disk via ``joblib``.
* ``apply_delta_calibrator`` – Given a delta and the home moneyline odds,
  return a calibrated home win probability.

The calibrator is stored as a dictionary mapping bucket names to fitted
``IsotonicRegression`` instances.  Buckets are determined by the sign and
magnitude of American odds (e.g. favourite/dog and small/medium/big).
``fallback`` is reserved for future extensions (e.g. global calibrator).

Example usage:

    from src.model.market_relative_calibration import (
        fit_delta_calibrator, save_delta_calibrator, load_delta_calibrator,
        apply_delta_calibrator,
    )

    # Train calibrator from per-game historical data
    calibrator = fit_delta_calibrator(df)
    save_delta_calibrator(calibrator, "artifacts/delta_calibrator.joblib")

    # Later, load and apply calibrator
    calib = load_delta_calibrator("artifacts/delta_calibrator.joblib")
    p_cal = apply_delta_calibrator(delta=0.05, ml_home=-150, calibrator=calib)

"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from typing import Dict, Optional

from src.model.calibration import fit_isotonic, apply_calibrator

###############################################################################
# Bucketing logic
###############################################################################

def _bucket_from_odds(odds: Optional[float]) -> Optional[str]:
    """Compute a bucket label from American odds.

    We divide odds into favourite vs underdog and then into magnitude bands.
    If odds is None or invalid, return None (which callers should handle).

    Buckets:
        fav_small : -100 >= odds > -200
        fav_medium: -200 >= odds > -400
        fav_big   : odds <= -400
        dog_small : 100 <= odds < 200
        dog_medium: 200 <= odds < 400
        dog_big   : odds >= 400

    Args:
        odds: American moneyline odds (signed value, >=100 or <=-100).

    Returns:
        Bucket name string or None.
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if np.isnan(o) or np.isinf(o) or abs(o) < 100:
        return None
    # Favourite if negative
    if o < 0:
        mag = abs(o)
        if mag <= 200:
            return "fav_small"
        if mag <= 400:
            return "fav_medium"
        return "fav_big"
    # Underdog if positive
    else:
        mag = o
        if mag < 200:
            return "dog_small"
        if mag < 400:
            return "dog_medium"
        return "dog_big"


###############################################################################
# Training and saving calibrators
###############################################################################

def fit_delta_calibrator(df: pd.DataFrame, *, min_samples: int = 25) -> Dict[str, object]:
    """Fit per-bucket isotonic calibrators on model vs market probability delta.

    Args:
        df: DataFrame with columns ``model_prob_home_raw``, ``market_prob_home``,
            ``home_win_actual``, and ``ml_home_consensus``.
        min_samples: Minimum number of samples required to fit a calibrator for a
            bucket.  Buckets with fewer samples will be skipped.

    Returns:
        Dictionary with keys:
            'calibrators': mapping bucket -> IsotonicRegression
            'min_samples': min_samples used
    """
    required_cols = {"model_prob_home_raw", "market_prob_home", "home_win_actual", "ml_home_consensus"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"[delta_calibrator] Missing required columns: {missing}")

    # Compute delta and bucket
    df = df.copy()
    df["model_prob_home_raw"] = pd.to_numeric(df["model_prob_home_raw"], errors="coerce")
    df["market_prob_home"] = pd.to_numeric(df["market_prob_home"], errors="coerce")
    df["home_win_actual"] = pd.to_numeric(df["home_win_actual"], errors="coerce")
    df["ml_home_consensus"] = pd.to_numeric(df["ml_home_consensus"], errors="coerce")
    df["delta"] = df["model_prob_home_raw"] - df["market_prob_home"]
    df["bucket"] = df["ml_home_consensus"].apply(_bucket_from_odds)

    calibrators: Dict[str, object] = {}

    for bucket, group in df.groupby("bucket"):
        if bucket is None:
            continue
        # Drop rows with NaNs
        mask = (~group["delta"].isna()) & (~group["home_win_actual"].isna())
        sub = group.loc[mask, ["delta", "home_win_actual"]]
        if len(sub) < min_samples:
            # Not enough samples; skip bucket
            continue
        # Fit isotonic regression calibrator on delta -> win probability
        calib = fit_isotonic(sub["home_win_actual"].values, sub["delta"].values)
        calibrators[bucket] = calib

    return {"calibrators": calibrators, "min_samples": min_samples}


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

def apply_delta_calibrator(delta: Optional[float], ml_home_odds: Optional[float], calibrator: Dict[str, object]) -> Optional[float]:
    """Apply a delta calibrator to produce a calibrated home win probability.

    Args:
        delta: The difference ``p_model - p_market``.
        ml_home_odds: The home moneyline odds used to determine bucket.
        calibrator: Dictionary returned by ``fit_delta_calibrator`` or
            loaded via ``load_delta_calibrator``.

    Returns:
        Calibrated home win probability in [0, 1], or None if calibration is
        unavailable for the given bucket or inputs are invalid.
    """
    if delta is None or ml_home_odds is None:
        return None
    try:
        d = float(delta)
        o = float(ml_home_odds)
    except Exception:
        return None
    # Guard against NaNs or values outside (0,1) for p_market or p_model
    if np.isnan(d) or np.isinf(d):
        return None
    bucket = _bucket_from_odds(o)
    calibrators = calibrator.get("calibrators", {})
    if bucket is None or bucket not in calibrators:
        return None
    iso = calibrators[bucket]
    # Use the delta as the 'probability' input for isotonic.
    # We must clip the delta into a small range to avoid extreme extrapolation.
    # Clip delta to [-1, 1] (model probs minus market probs cannot exceed this).
    d_clipped = max(min(d, 1.0), -1.0)
    # Normalise delta to [0,1] as isotonic expects probabilities.
    # We'll map -1 -> 0, +1 -> 1.
    x = (d_clipped + 1.0) / 2.0
    p_cal = apply_calibrator(np.array([x]), iso)[0]
    # Ensure within [0,1]
    return max(0.0, min(1.0, float(p_cal)))
