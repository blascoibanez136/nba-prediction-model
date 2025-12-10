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

Assumed inputs
--------------
1) Model predictions CSV (from model.predict), with at least:
    - game_id            (string or int; unique per game)
    - fair_spread        (model-derived fair spread; negative = home favorite)
    - fair_total         (model-derived fair total points)
    - home_win_prob      (model-derived home win probability in [0, 1])

2) Normalized odds CSV (CLOSE snapshot), with a *line-level* schema, e.g.:
    - game_id            (matches predictions)
    - book               (e.g. 'dk', 'fd', 'mgm', ...)
    - market             ('spreads', 'totals', 'h2h')
    - outcome            (for spreads/h2h: 'home' or 'away';
                          for totals: 'over' or 'under')
    - price              (American odds, e.g. -110, +105)
    - point              (spread / total line as float)

If your odds_normalizer produced slightly different column names,
modify the column mappings at the top accordingly.

Usage (imported)
----------------
    from src.model.market_ensemble import apply_market_ensemble

    blended_df = apply_market_ensemble(
        preds_df,
        odds_close_df,
    )

Usage (CLI)
-----------
    python -m src.model.market_ensemble \
        --predictions-csv outputs/predictions_2024-02-05.csv \
        --odds-close-csv data/_snapshots/odds_close_2024-02-05_normalized.csv \
        --out-csv outputs/predictions_2024-02-05_market.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurable column mappings
# ---------------------------------------------------------------------------

@dataclass
class PredictionsCols:
    game_id: str = "game_id"
    fair_spread: str = "fair_spread"
    fair_total: str = "fair_total"
    home_win_prob: str = "home_win_prob"


@dataclass
class OddsCols:
    game_id: str = "game_id"
    book: str = "book"
    market: str = "market"
    outcome: str = "outcome"
    price: str = "price"   # American odds
    point: str = "point"   # spread / total point


PRED_COLS = PredictionsCols()
ODDS_COLS = OddsCols()


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
        containing spreads, totals, and moneylines (h2h).

    Returns
    -------
    market_features : pd.DataFrame
        Columns:
            - game_id
            - consensus_close_spread_home
            - spread_dispersion
            - consensus_close_total
            - total_dispersion
            - market_implied_home_win_prob
    """
    g = ODDS_COLS

    # Defensive copy
    df = odds_df.copy()

    # Ensure required columns exist
    required_cols = [g.game_id, g.book, g.market, g.outcome, g.price, g.point]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Odds DataFrame is missing required columns: {missing}")

    # ----- SPREADS -----
    spreads = df[df[g.market] == "spreads"].copy()
    # We'll treat 'home' outcome as "home spread" (point usually negative if fav)
    home_spreads = spreads[spreads[g.outcome] == "home"]

    spread_agg = (
        home_spreads
        .groupby(g.game_id)[g.point]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "consensus_close_spread_home",
                "std": "spread_dispersion",
            }
        )
    )

    # ----- TOTALS -----
    totals = df[df[g.market] == "totals"].copy()
    # Use 'over' lines for consensus total
    overs = totals[totals[g.outcome] == "over"]

    total_agg = (
        overs
        .groupby(g.game_id)[g.point]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "consensus_close_total",
                "std": "total_dispersion",
            }
        )
    )

    # ----- MONEYLINES (H2H) -----
    h2h = df[df[g.market] == "h2h"].copy()
    home_ml = h2h[h2h[g.outcome] == "home"].copy()

    home_ml["implied_prob"] = home_ml[g.price].astype(float).apply(_american_to_implied_prob)

    ml_agg = (
        home_ml
        .groupby(g.game_id)["implied_prob"]
        .mean()
        .to_frame("market_implied_home_win_prob")
    )

    # Combine
    market_features = (
        spread_agg.join(total_agg, how="outer")
        .join(ml_agg, how="outer")
        .reset_index()
        .rename(columns={g.game_id: PRED_COLS.game_id})
    )

    return market_features


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

    Implementation:
        - We compute a logistic function of (dispersion - pivot) so that:
            dispersion << pivot  -> model_weight ~ min_weight (market heavy)
            dispersion >> pivot  -> model_weight ~ max_weight (model heavy)
    """
    if dispersion is None or np.isnan(dispersion):
        # If we don't have dispersion info, fallback to a neutral weight.
        return 0.50

    # logistic on (dispersion - pivot) then flipped
    z = sharpness * (dispersion - pivot)
    base = _sigmoid(z)  # increases with dispersion
    # base ~ 0 when dispersion << pivot
    # base ~ 1 when dispersion >> pivot

    # Map base in [0,1] -> [min_weight, max_weight]
    return min_weight + base * (max_weight - min_weight)


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
    except ValueError:
        # if either prob is 0 or 1 etc., fall back to simple convex combo
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
            - fair_spread
            - fair_total
            - home_win_prob
    odds_close_df : pd.DataFrame
        Normalized CLOSE odds snapshot, line-level (spreads, totals, h2h).
    spread_dispersion_pivot : float
        Pivot for dispersion -> weight mapping for spreads. Smaller pivot
        means we treat even small disagreements as "high dispersion".
    total_dispersion_pivot : float
        Same for totals.

    Returns
    -------
    blended_df : pd.DataFrame
        Original preds_df plus:
            - consensus_close_spread_home
            - spread_dispersion
            - consensus_close_total
            - total_dispersion
            - market_implied_home_win_prob
            - fair_spread_market
            - fair_total_market
            - home_win_prob_market
    """
    p = PRED_COLS

    if p.game_id not in preds_df.columns:
        raise ValueError(f"Predictions DataFrame missing '{p.game_id}' column.")
    if p.fair_spread not in preds_df.columns:
        raise ValueError(f"Predictions DataFrame missing '{p.fair_spread}' column.")
    if p.fair_total not in preds_df.columns:
        raise ValueError(f"Predictions DataFrame missing '{p.fair_total}' column.")
    if p.home_win_prob not in preds_df.columns:
        raise ValueError(f"Predictions DataFrame missing '{p.home_win_prob}' column.")

    preds = preds_df.copy()

    # Compute per-game market features from odds
    market_features = compute_market_features(odds_close_df)

    # Merge
    merged = preds.merge(
        market_features,
        on=p.game_id,
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

    # For win probability, we can reuse spread_weights or a neutral 0.5 if desired.
    prob_weights = spread_weights

    # Line blending
    blended_spreads = []
    blended_totals = []
    blended_probs = []

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
        help="Path to model predictions CSV (from model.predict).",
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
