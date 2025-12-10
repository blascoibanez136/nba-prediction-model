"""
Market-aware ensemble for NBA Pro-Lite.

Responsibilities
----------------
- Read normalized odds snapshot CSVs (typically CLOSE) produced by odds_normalizer.py
- Compute per-game market features:
    * consensus_close_spread_home
    * consensus_close_total
    * spread_dispersion
    * total_dispersion
    * market_implied_home_win_prob (from moneylines)
- Blend model predictions with market information to produce:
    * fair_spread_market
    * fair_total_market
    * home_win_prob_market

Join key
--------
We join model predictions to market features using:

    merge_key = "{home_team}__{away_team}__{game_date}"

where:
    - home_team / away_team are lowercased, stripped
    - game_date is the NBA schedule date (matching America/New_York conversion
      in odds_normalizer.py)

This avoids the mismatch between:
    - balldontlie numeric game_id
    - The Odds API opaque hash game_id
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurable column mappings
# ---------------------------------------------------------------------------

@dataclass
class PredictionsCols:
    game_id: str = "game_id"
    game_date: str = "game_date"
    merge_key: str = "merge_key"
    fair_spread: str = "fair_spread"
    fair_total: str = "fair_total"
    home_win_prob: str = "home_win_prob"


@dataclass
class OddsCols:
    game_id: str = "game_id"
    merge_key: str = "merge_key"
    book: str = "book"
    snapshot_type: str = "snapshot_type"
    game_date: str = "game_date"
    home_team: str = "home_team"
    away_team: str = "away_team"
    ml_home: str = "ml_home"
    ml_away: str = "ml_away"
    spread_home_point: str = "spread_home_point"
    spread_home_price: str = "spread_home_price"
    spread_away_point: str = "spread_away_point"
    spread_away_price: str = "spread_away_price"
    total_point: str = "total_point"
    total_over_price: str = "total_over_price"
    total_under_price: str = "total_under_price"


PRED_COLS = PredictionsCols()
ODDS_COLS = OddsCols()


def _norm_key(name: str) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


def _make_merge_key(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    date_col: str,
    out_col: str = "merge_key",
) -> pd.Series:
    def mk(row):
        return f"{_norm_key(row[home_col])}__{_norm_key(row[away_col])}__{row[date_col]}"
    return df.apply(mk, axis=1)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-6) -> float:
    p_clipped = min(max(p, eps), 1.0 - eps)
    return math.log(p_clipped / (1.0 - p_clipped))


def _inv_logit(z: float) -> float:
    return _sigmoid(z)


def _american_to_implied_prob(odds: float) -> Optional[float]:
    """
    Convert American odds to implied probability (no vig removed).
    Returns None if odds is NaN or 0.
    """
    if odds is None or np.isnan(odds) or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


# ---------------------------------------------------------------------------
# Market feature computation
# ---------------------------------------------------------------------------

def compute_market_features(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game market features from normalized odds.

    Parameters
    ----------
    odds_df : pd.DataFrame
        Normalized odds for a *single* snapshot (typically CLOSE),
        containing spreads, totals, and moneylines per (game, book).

    Returns
    -------
    market_features : pd.DataFrame
        Columns:
            - merge_key
            - consensus_close_spread_home
            - spread_dispersion
            - consensus_close_total
            - total_dispersion
            - market_implied_home_win_prob
    """
    g = ODDS_COLS

    df = odds_df.copy()

    # Ensure a merge_key exists on odds side
    if g.merge_key not in df.columns:
        if g.home_team not in df.columns or g.away_team not in df.columns or g.game_date not in df.columns:
            raise ValueError(
                "Odds DataFrame missing merge_key and "
                "cannot recompute it (need home_team, away_team, game_date)."
            )
        df[g.merge_key] = _make_merge_key(df, g.home_team, g.away_team, g.game_date, g.merge_key)

    # --- Spreads (home perspective) ---
    if g.spread_home_point not in df.columns:
        df[g.spread_home_point] = np.nan

    spread_agg = (
        df.groupby(g.merge_key)[g.spread_home_point]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "consensus_close_spread_home",
                "std": "spread_dispersion",
            }
        )
    )

    # --- Totals (Over side) ---
    if g.total_point not in df.columns:
        df[g.total_point] = np.nan

    total_agg = (
        df.groupby(g.merge_key)[g.total_point]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "consensus_close_total",
                "std": "total_dispersion",
            }
        )
    )

    # --- Moneylines (home side implied probability) ---
    ml_probs = []
    for val in df.get(g.ml_home, []):
        if pd.isna(val):
            ml_probs.append(np.nan)
        else:
            ml_probs.append(_american_to_implied_prob(float(val)))
    df["home_ml_implied_prob"] = ml_probs

    ml_agg = (
        df.groupby(g.merge_key)["home_ml_implied_prob"]
        .mean()
        .to_frame("market_implied_home_win_prob")
    )

    market_features = (
        spread_agg.join(total_agg, how="outer")
        .join(ml_agg, how="outer")
        .reset_index()
    )

    return market_features  # has merge_key as a column


# ---------------------------------------------------------------------------
# Weighting & blending logic
# ---------------------------------------------------------------------------

def _compute_dispersion_weight(
    dispersion: float,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
    pivot: float = 1.5,
    sharpness: float = 2.0,
) -> float:
    """
    Convert a dispersion value (std dev of lines across books) into a model weight.

    Intuition:
        - Low dispersion -> market in strong agreement -> trust market more
        - High dispersion -> market disagrees -> trust model more
    """
    if dispersion is None or np.isnan(dispersion):
        # If we don't have dispersion info, fallback to a neutral weight.
        return 0.50

    z = sharpness * (dispersion - pivot)
    base = _sigmoid(z)  # increases with dispersion

    # Map base in [0,1] -> [min_weight, max_weight]
    return min_weight + base * (max_weight - max_weight * 0 + base * 0) - base * min_weight if False else \
        min_weight + base * (max_weight - min_weight)


def blend_lines(
    model_line: Optional[float],
    market_line: Optional[float],
    model_weight: float,
) -> Optional[float]:
    """
    Blend a model line and a market line given a model weight.

    If market_line is missing, fallback to model_line.
    If model_line is missing, fallback to market_line.
    """
    if model_line is None or np.isnan(model_line):
        return market_line
    if market_line is None or np.isnan(market_line):
        return model_line

    w = float(model_weight)
    w = max(0.0, min(1.0, w))
    return w * model_line + (1.0 - w) * market_line


def blend_probs(
    model_prob: Optional[float],
    market_prob: Optional[float],
    model_weight: float,
) -> Optional[float]:
    """
    Blend a model probability with a market probability using logit space.

    If market_prob is missing, return model_prob.
    If model_prob is missing, return market_prob.
    """
    if model_prob is None or np.isnan(model_prob):
        return market_prob
    if market_prob is None or np.isnan(market_prob):
        return model_prob

    w = float(model_weight)
    w = max(0.0, min(1.0, w))

    try:
        z_model = _logit(float(model_prob))
        z_market = _logit(float(market_prob))
    except Exception:
        # If either prob is pathological, fall back to simple convex combo
        return w * model_prob + (1 - w) * market_prob

    z_blend = w * z_model + (1.0 - w) * z_market
    return _inv_logit(z_blend)


# ---------------------------------------------------------------------------
# Main ensemble application
# ---------------------------------------------------------------------------

def apply_market_ensemble(
    preds_df: pd.DataFrame,
    odds_close_df: pd.DataFrame,
    spread_dispersion_pivot: float = 1.5,
    total_dispersion_pivot: float = 5.0,
) -> pd.DataFrame:
    """
    Merge model predictions with market odds and compute market-aware outputs.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Model predictions with columns:
            - game_id
            - game_date
            - merge_key
            - fair_spread
            - fair_total
            - home_win_prob
    odds_close_df : pd.DataFrame
        Normalized CLOSE odds snapshot, book-level.
    """
    p = PRED_COLS
    g = ODDS_COLS

    for col in (p.game_id, p.game_date, p.fair_spread, p.fair_total, p.home_win_prob):
        if col not in preds_df.columns:
            raise ValueError(f"Predictions DataFrame missing '{col}' column.")

    preds = preds_df.copy()
    odds = odds_close_df.copy()

    # Ensure merge_key exists on preds side (it should already be there from feature_builder)
    if p.merge_key not in preds.columns:
        preds[p.merge_key] = _make_merge_key(preds, "home_team", "away_team", p.game_date, p.merge_key)

    # Compute per-game market features from odds (keyed by merge_key)
    market_features = compute_market_features(odds)

    # Merge on merge_key (shared across model + odds)
    merged = preds.merge(
        market_features,
        left_on=p.merge_key,
        right_on=g.merge_key,
        how="left",
        validate="one_to_one",
    )

    # Compute weights for spreads/totals separately
    spread_weights = merged["spread_dispersion"].apply(
        lambda d: _compute_dispersion_weight(
            d,
            min_weight=0.20,
            max_weight=0.80,
            pivot=spread_dispersion_pivot,
            sharpness=2.0,
        )
    )

    total_weights = merged["total_dispersion"].apply(
        lambda d: _compute_dispersion_weight(
            d,
            min_weight=0.20,
            max_weight=0.80,
            pivot=total_dispersion_pivot,
            sharpness=2.0,
        )
    )

    # For win probability, we can reuse spread_weights
    prob_weights = spread_weights

    blended_spreads: List[float] = []
    blended_totals: List[float] = []
    blended_probs: List[float] = []

    for i, row in merged.iterrows():
        # Spread
        model_spread = row[p.fair_spread]
        market_spread = row.get("consensus_close_spread_home", np.nan)
        w_spread = spread_weights.iat[i]
        blended_spreads.append(
            blend_lines(model_spread, market_spread, w_spread)
        )

        # Total
        model_total = row[p.fair_total]
        market_total = row.get("consensus_close_total", np.nan)
        w_total = total_weights.iat[i]
        blended_totals.append(
            blend_lines(model_total, market_total, w_total)
        )

        # Win probability
        model_prob = row[p.home_win_prob]
        market_prob = row.get("market_implied_home_win_prob", np.nan)
        w_prob = prob_weights.iat[i]
        blended_probs.append(
            blend_probs(model_prob, market_prob, w_prob)
        )

    merged["fair_spread_market"] = blended_spreads
    merged["fair_total_market"] = blended_totals
    merged["home_win_prob_market"] = blended_probs

    # Aliases for existing downstream code (edge_picker, etc.)
    if "consensus_close_spread_home" in merged.columns and "consensus_close" not in merged.columns:
        merged["consensus_close"] = merged["consensus_close_spread_home"]
    if "spread_dispersion" in merged.columns and "book_dispersion" not in merged.columns:
        merged["book_dispersion"] = merged["spread_dispersion"]

    return merged


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply market-aware ensemble to NBA Pro-Lite predictions."
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Path to model predictions CSV (from model/predict.py).",
    )
    parser.add_argument(
        "--odds-close-csv",
        required=True,
        help="Path to normalized CLOSE odds CSV (from odds_normalizer.py).",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output path for blended predictions CSV.",
    )
    parser.add_argument(
        "--spread-dispersion-pivot",
        type=float,
        default=1.5,
        help="Pivot for dispersion->weight mapping (spreads).",
    )
    parser.add_argument(
        "--total-dispersion-pivot",
        type=float,
        default=5.0,
        help="Pivot for dispersion->weight mapping (totals).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    preds_df = pd.read_csv(args.predictions_csv)
    odds_close_df = pd.read_csv(args.odds_close_csv)

    blended_df = apply_market_ensemble(
        preds_df,
        odds_close_df,
        spread_dispersion_pivot=args.spread_dispersion_pivot,
        total_dispersion_pivot=args.total_dispersion_pivot,
    )

    blended_df.to_csv(args.out_csv, index=False)
    print(
        f"[market_ensemble] Wrote blended predictions with market features to {args.out_csv} "
        f"({len(blended_df)} rows)."
    )


if __name__ == "__main__":
    main()
