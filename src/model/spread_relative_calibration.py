"""
Market-relative spread (ATS) calibration.

We calibrate in residual space:
  residual = fair_spread_model - home_spread_consensus

Interpretation:
  residual > 0  => model believes home will outperform market line (home ATS)
  residual < 0  => model believes away will outperform market line (away ATS)

This module provides:
- load_spread_calibrator(path)
- apply_spread_calibrator(residual, home_spread_consensus, calibrator) -> P(home_covers)

Calibrator format (joblib):
{
  "type": "spread_isotonic_bucketed_v1",
  "global": <IsotonicRegression>,
  "buckets": [{"name":..., "lo":..., "hi":..., "model": <IsotonicRegression or None>} ...],
  "meta": {...}
}
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import joblib


def load_spread_calibrator(path: str) -> Dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "type" not in obj:
        raise RuntimeError(f"[spread_cal] Invalid calibrator object at {path}")
    if obj.get("type") != "spread_isotonic_bucketed_v1":
        raise RuntimeError(
            f"[spread_cal] Unexpected calibrator type={obj.get('type')} (expected spread_isotonic_bucketed_v1)"
        )
    if "global" not in obj or "buckets" not in obj:
        raise RuntimeError("[spread_cal] Calibrator missing required keys: global, buckets")
    return obj


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _bucket_name(abs_spread: float, buckets) -> str:
    for b in buckets:
        lo = float(b["lo"])
        hi = float(b["hi"])
        if abs_spread >= lo and abs_spread < hi:
            return b["name"]
    # fallthrough to last
    return buckets[-1]["name"] if buckets else "global"


def apply_spread_calibrator(
    residual: Optional[float],
    home_spread_consensus: Optional[float],
    calibrator: Dict[str, Any],
    clip_eps: float = 1e-4,
) -> Optional[float]:
    """
    Returns calibrated P(home_covers) in (0,1), or None if cannot compute.
    """
    r = _to_float(residual)
    s = _to_float(home_spread_consensus)
    if r is None or s is None:
        return None

    abs_spread = abs(s)
    buckets = calibrator.get("buckets", [])
    name = _bucket_name(abs_spread, buckets)

    # pick bucket model if present, else global
    model = None
    for b in buckets:
        if b.get("name") == name:
            model = b.get("model", None)
            break
    if model is None:
        model = calibrator.get("global", None)

    if model is None:
        return None

    try:
        p = float(model.predict([r])[0])
    except Exception:
        return None

    # hard clip to avoid 0/1 which can explode EV logic
    p = max(clip_eps, min(1.0 - clip_eps, p))
    return p
