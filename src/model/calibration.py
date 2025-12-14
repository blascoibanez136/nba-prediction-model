"""
Calibration utilities for win-probability models.

This module provides a simple interface for fitting an isotonic regression
calibration function on predicted probabilities and their corresponding
binary outcomes.  It also provides helper functions to apply the fitted
calibrator and to save/load calibrators using ``joblib``.

The purpose of calibration is to correct systematic overconfidence or
underconfidence in probability estimates.  A well‑calibrated model will
output probabilities that reflect the true frequency of the event.

Example:

    from src.model.calibration import fit_isotonic, apply_calibrator, save_calibrator

    # Fit calibrator
    calibrator = fit_isotonic(y_true, y_pred)
    save_calibrator(calibrator, "models/calibrator.joblib")

    # Later, apply calibrator to new probabilities
    calibrated = apply_calibrator(probs, calibrator)
"""

from __future__ import annotations

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic regression calibrator.

    Args:
        y_true: Array of true binary outcomes (0 or 1).
        y_pred: Array of predicted probabilities (floats in [0, 1]).

    Returns:
        A fitted ``IsotonicRegression`` instance that maps input probabilities
        to calibrated probabilities.  Predictions outside the range of the
        fitted data will be clipped to the nearest boundary.
    """
    # Convert to numpy arrays and ensure float type
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Drop NaNs
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Clip probabilities to avoid issues at the boundaries
    eps = 1e-6
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    # Instantiate and fit isotonic regression (increasing function)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred, y_true)
    return calibrator


def apply_calibrator(probs: np.ndarray, calibrator: IsotonicRegression) -> np.ndarray:
    """Apply a fitted calibrator to an array of probabilities.

    Args:
        probs: Array of predicted probabilities.
        calibrator: A fitted ``IsotonicRegression`` instance.

    Returns:
        An array of calibrated probabilities with the same shape as ``probs``.
    """
    probs = np.asarray(probs, dtype=float)
    # Clip to [0, 1] and avoid extremes
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    return calibrator.predict(probs)


def save_calibrator(calibrator: IsotonicRegression, path: str) -> None:
    """Serialize a calibrator to disk using joblib.

    Args:
        calibrator: Fitted ``IsotonicRegression`` instance.
        path: Destination file path (typically ``.joblib``).
    """
    joblib.dump(calibrator, path)


def load_calibrator(path: str) -> IsotonicRegression:
    """Load a calibrator from disk.

    Args:
        path: File path to a saved calibrator.

    Returns:
        Loaded ``IsotonicRegression`` instance.
    """
    return joblib.load(path)
