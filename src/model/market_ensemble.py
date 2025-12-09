"""
Market-aware ensemble for NBA Pro-Lite.

Takes:
- model-based predictions (our side)
    - home_win_prob
    - fair_spread        (home side, negative if home is favored)
    - fair_total         (optional)
- market features (schedule/odds join)
    - consensus_close    (spread from odds, home side)
    - book_dispersion    (std dev of spreads across books)
    - open_consensus     (optional, used for movement)
    - close_consensus    (optional, used for movement)
    - line_move          (optional, close_consensus - open_consensus)

Core function:
    _apply_market_ensemble_core(df)

Public API (backwards compatible):
    apply_market_ensemble(preds, odds=None)

- If `odds` is provided and has merge_key, it will be LEFT-JOINed into preds.
- If `odds` is None, we just use preds as-is (model-only ensemble).
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd


# -------------------------------------------------------------------------
# Numeric helpers
# -------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-6) -> float:
    """Safe logit transform with clamping."""
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1.0 - p))


def _blend(a: float, b: float, w: float) -> float:
    """Blend between a and b with weight on b."""
    return (1.0 - w) * a + w * b


# -------------------------------------------------------------------------
# Core market logic
# -------------------------------------------------------------------------

def _market_spread_to_prob(spread: float, scale: float = 0.12) -> float:
    """
    Rough mapping from spread (home side) to home win probability.

    We don't need perfection here; just a reasonable monotonic relationship:
      - 0 spread ~ 50%
      - -3.5 favorite ~ 60–65%
      - -7.0 favorite ~ 70–75%
    """
    return _sigmoid(-spread * scale)


def _dispersion_to_weight(disp: Optional[float]) -> float:
    """
    Convert book dispersion to a market weight in [0.0, 0.9].

    Intuition:
      - If books are very aligned (disp ~ 0.0), trust market heavily (w_market ~ 0.8–0.9)
      - If books wildly disagree (disp >= 3.0), rely mostly on model (w_market ~ 0.2)
      - Missing dispersion -> low trust in market
    """
    if disp is None or pd.isna(disp):
        return 0.2

    # Clamp dispersion to [0, 3]
    d = max(0.0, min(3.0, float(disp)))
    # Linear map: d=0 -> 0.85, d=3 -> 0.2
    hi, lo = 0.85, 0.2
    w = hi - (hi - lo) * (d / 3.0)
    return max(lo, min(hi, w))


def _movement_to_logit_shift(line_move: Optional[float], coeff: float = 0.10) -> float:
    """
    Convert line movement into a small logit shift.

    A movement of 1 point in the spread is meaningful but not massive.
    We use a small coefficient so that e.g. 1 point of steam nudges
    probability by a few percentage points.
    """
    if line_move is None or pd.isna(line_move):
        return 0.0
    return float(line_move) * coeff


def _apply_market_ensemble_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core implementation working on a single dataframe that already includes
    both model and market columns.

    Expected input columns (as available):
      - home_win_prob        [REQUIRED]
      - fair_spread          [REQUIRED]
      - fair_total           [OPTIONAL]
      - consensus_close      [OPTIONAL]
      - book_dispersion      [OPTIONAL]
      - open_consensus       [OPTIONAL]
      - close_consensus      [OPTIONAL]
      - line_move            [OPTIONAL]

    Output columns added:
      - fair_spread_market
      - home_win_prob_market
      - fair_total_market
    """
    df = df.copy()

    if "home_win_prob" not in df.columns:
        raise ValueError("_apply_market_ensemble_core: 'home_win_prob' is required.")
    if "fair_spread" not in df.columns:
        raise ValueError("_apply_market_ensemble_core: 'fair_spread' is required.")

    fair_spread_mkt = []
    home_prob_mkt = []
    fair_total_mkt = []

    for _, row in df.iterrows():
        p_model = row["home_win_prob"]
        fair_spread = row["fair_spread"]
        fair_total = row.get("fair_total", None)

        consensus = row.get("consensus_close", None)
        dispersion = row.get("book_dispersion", None)
        line_move = row.get("line_move", None)

        # If model prob is missing, fall back to market or 0.5
        if pd.isna(p_model):
            if consensus is None or pd.isna(consensus):
                p_model = 0.5
            else:
                p_model = _market_spread_to_prob(float(consensus))

        # If no market info at all, just copy model values through.
        if consensus is None or pd.isna(consensus):
            fair_spread_mkt.append(fair_spread)
            home_prob_mkt.append(p_model)
            fair_total_mkt.append(fair_total if fair_total is not None else None)
            continue

        # Market-implied win probability
        try:
            p_market = _market_spread_to_prob(float(consensus))
        except Exception:
            fair_spread_mkt.append(fair_spread)
            home_prob_mkt.append(p_model)
            fair_total_mkt.append(fair_total if fair_total is not None else None)
            continue

        w_market = _dispersion_to_weight(dispersion)
        logit_model = _logit(float(p_model))
        logit_market = _logit(float(p_market))

        # Movement-based logit adjustment (steam)
        logit_shift = _movement_to_logit_shift(line_move)

        # Blend in logit space, then apply movement shift
        logit_blend = _blend(logit_model, logit_market, w_market) + logit_shift
        p_final = _sigmoid(logit_blend)

        # Blend spreads as well
        try:
            fair_spread_blend = _blend(float(fair_spread), float(consensus), w_market)
        except Exception:
            fair_spread_blend = fair_spread

        # Totals: pass through model fair_total for now
        fair_total_blend = fair_total if fair_total is not None else None

        fair_spread_mkt.append(fair_spread_blend)
        home_prob_mkt.append(p_final)
        fair_total_mkt.append(fair_total_blend)

    df["fair_spread_market"] = fair_spread_mkt
    df["home_win_prob_market"] = home_prob_mkt
    df["fair_total_market"] = fair_total_mkt

    return df


# -------------------------------------------------------------------------
# Public API (backwards compatible)
# -------------------------------------------------------------------------

def apply_market_ensemble(preds: pd.DataFrame,
                          odds: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Backwards-compatible wrapper.

    Old signature (still supported):
        apply_market_ensemble(preds, odds)

    New recommended usage:
        merged = preds.merge(odds_features, on="merge_key", how="left")
        out = apply_market_ensemble(merged)

    Behavior:
      - If `odds` is provided and non-empty, and both dataframes
        have a 'merge_key' column, we LEFT-JOIN odds into preds.
      - If `odds` is None or empty, we use preds as-is and the core
        ensemble will effectively be model-only when market columns
        are missing.
    """
    if odds is not None and not odds.empty and "merge_key" in preds.columns and "merge_key" in odds.columns:
        merged = preds.merge(odds, on="merge_key", how="left")
    else:
        merged = preds

    return _apply_market_ensemble_core(merged)
