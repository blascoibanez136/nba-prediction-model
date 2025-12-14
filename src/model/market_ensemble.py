"""
Market-aware ensemble for NBA Pro-Lite / Elite.

Blends model-based predictions with betting market information.

SUPPORTED ODDS FORMATS
----------------------
1) NORMALIZED FORMAT
   merge_key, market, side, point, price, snapshot_type

2) WIDE SNAPSHOT FORMAT
   merge_key, book,
   ml_home, ml_away,
   spread_home_point / spread_home / spread,
   total_point / total,
   snapshot_type

OUTPUTS (backtest-ready)
------------------------
- home_win_prob_model
- home_win_prob_market
- home_win_prob (blended)
- fair_spread_model
- fair_spread_market
- fair_spread (blended)
- fair_total_model
- fair_total_market
- fair_total (blended)
- market_weight
- home_spread_dispersion

Backward-compatibility aliases:
- consensus_close   -> home_spread_consensus
- book_dispersion   -> home_spread_dispersion
- consensus_total   -> total_consensus
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Basic math utilities
# ---------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def american_to_prob(odds: float) -> Optional[float]:
    """Convert American moneyline odds to implied probability.

    Returns None if odds are missing or zero.
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0 or math.isnan(o):
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


# ---------------------------------------------------------------------
# Spread -> win probability
# ---------------------------------------------------------------------

def spread_to_win_prob(spread_home: float, slope: float = -0.165) -> Optional[float]:
    """Map a point spread to a win probability using a logistic curve.

    A negative slope indicates larger spreads favour the road team.
    Returns None if spread is missing.
    """
    if spread_home is None:
        return None
    try:
        s = float(spread_home)
    except (TypeError, ValueError):
        return None
    if math.isnan(s):
        return None
    return _sigmoid(slope * s)


# ---------------------------------------------------------------------
# Dispersion -> market weight
# ---------------------------------------------------------------------

def compute_dispersion_weight(
    dispersion: Optional[float],
    *,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
    pivot: float = 1.5,
    sharpness: float = 2.0,
) -> float:
    """Map spread dispersion to a blending weight between model and market.

    A lower dispersion implies greater market consensus and a higher weight.
    Returns 0.5 if dispersion is missing or invalid.
    """
    if dispersion is None:
        return 0.5
    try:
        d = float(dispersion)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(d):
        return 0.5
    z = -sharpness * (d - pivot)
    base = _sigmoid(z)
    return float(min_weight + base * (max_weight - min_weight))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Safely cast specified columns to numeric, coercing errors to NaN."""
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------
# Normalized odds aggregation
# ---------------------------------------------------------------------

def _aggregate_from_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized odds into consensus statistics.

    Expects columns: merge_key, market, side, point, price. The `side` column
    indicates 'home' or 'away' for h2h moneylines and spreads, and 'over'
    for totals. Prices are American odds. Points are spread or total values.
    """
    df = df.copy()
    df["market"] = df["market"].astype(str).str.lower()
    df["side"] = df["side"].astype(str).str.lower()
    df["point"] = pd.to_numeric(df.get("point"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")

    # filter by market/side for spreads and totals
    spreads = df[(df.market == "spreads") & (df.side == "home")]
    totals = df[(df.market == "totals") & (df.side == "over")]
    # include both home and away for h2h so we can de‑vig
    h2h = df[(df.market == "h2h") & (df.side.isin(["home", "away"]))]

    # consensus spread mean and dispersion
    if not spreads.empty:
        spread_stats = (
            spreads.groupby("merge_key")["point"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
            .reset_index()
        )
    else:
        spread_stats = pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    # consensus total mean
    if not totals.empty:
        totals_stats = (
            totals.groupby("merge_key")["point"]
            .mean()
            .rename("total_consensus")
            .reset_index()
        )
    else:
        totals_stats = pd.DataFrame(columns=["merge_key", "total_consensus"])

    # moneyline price and de‑vigged probability consensus
    if not h2h.empty:
        # implied probability per outcome from price
        h2h["prob"] = h2h["price"].apply(american_to_prob)
        # mean price by side
        ml_price = (
            h2h.pivot_table(index="merge_key", columns="side", values="price", aggfunc="mean")
            .rename(columns={"home": "ml_home_consensus", "away": "ml_away_consensus"})
            .reset_index()
        )
        # mean implied probability by side
        ml_prob = (
            h2h.pivot_table(index="merge_key", columns="side", values="prob", aggfunc="mean")
            .rename(columns={"home": "home_ml_prob_raw", "away": "away_ml_prob_raw"})
            .reset_index()
        )
        ml_stats = ml_price.merge(ml_prob, on="merge_key", how="outer")

        def _devig_row(r: pd.Series) -> Optional[float]:
            """De‑vig a pair of implied probs into a single home implied prob.
            If both home and away probs are present and sum > 0, return the
            normalized home prob. Otherwise return the home raw prob.
            """
            ph = r.get("home_ml_prob_raw")
            pa = r.get("away_ml_prob_raw")
            if pd.notna(ph) and pd.notna(pa) and (ph + pa) > 0:
                return ph / (ph + pa)
            return ph

        ml_stats["home_ml_prob_consensus"] = ml_stats.apply(_devig_row, axis=1)
        # drop the raw prob columns
        ml_stats = ml_stats.drop(columns=[c for c in ["home_ml_prob_raw", "away_ml_prob_raw"] if c in ml_stats.columns])
    else:
        ml_stats = pd.DataFrame(columns=["merge_key", "ml_home_consensus", "ml_away_consensus", "home_ml_prob_consensus"])

    # merge all components
    out = spread_stats.merge(totals_stats, on="merge_key", how="outer")
    out = out.merge(ml_stats, on="merge_key", how="outer")
    return out


# ---------------------------------------------------------------------
# Wide odds aggregation (AUTO-DETECT columns)
# ---------------------------------------------------------------------

def _aggregate_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wide snapshot odds into consensus statistics.

    Auto-detects column names for spreads, totals, and moneyline prices.
    """
    df = df.copy()

    # helper to find the first matching column from a set
    def _find(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    spread_col = _find(["spread_home_point", "spread_home", "spread"])
    total_col = _find(["total_point", "total"])
    ml_home_col = _find(["ml_home", "home_ml"])
    ml_away_col = _find(["ml_away", "away_ml"])

    # cast numeric columns safely
    df = _safe_to_numeric(df, [spread_col, total_col, ml_home_col, ml_away_col])

    # spread consensus and dispersion
    if spread_col:
        spread_stats = (
            df.groupby("merge_key")[spread_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
            .reset_index()
        )
    else:
        spread_stats = pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    # total consensus
    if total_col:
        totals_stats = (
            df.groupby("merge_key")[total_col]
            .mean()
            .rename("total_consensus")
            .reset_index()
        )
    else:
        totals_stats = pd.DataFrame(columns=["merge_key", "total_consensus"])

    # moneyline consensus
    if ml_home_col and ml_away_col:
        ml = df[["merge_key", ml_home_col, ml_away_col]].copy()
        # mean price consensus per side
        ml_price = (
            ml.groupby("merge_key")[[ml_home_col, ml_away_col]]
            .mean()
            .rename(columns={ml_home_col: "ml_home_consensus", ml_away_col: "ml_away_consensus"})
            .reset_index()
        )
        # compute implied probability per row and de‑vig
        def _devig(r: pd.Series) -> Optional[float]:
            ph = american_to_prob(r[ml_home_col])
            pa = american_to_prob(r[ml_away_col])
            # if both sides have probs and sum > 0, de‑vig; else return ph
            return ph / (ph + pa) if (ph is not None and pa is not None and (ph + pa) > 0) else ph
        ml["prob"] = ml.apply(_devig, axis=1)
        ml_prob = ml.groupby("merge_key")["prob"].mean().rename("home_ml_prob_consensus").reset_index()
        ml_stats = ml_price.merge(ml_prob, on="merge_key", how="outer")
    elif ml_home_col:
        ml = df[["merge_key", ml_home_col]].copy()
        ml_price = (
            ml.groupby("merge_key")[ml_home_col]
            .mean()
            .rename("ml_home_consensus")
            .reset_index()
        )
        # compute implied probability from home odds only (no de‑vig)
        ml["prob"] = ml[ml_home_col].apply(american_to_prob)
        ml_prob = ml.groupby("merge_key")["prob"].mean().rename("home_ml_prob_consensus").reset_index()
        ml_stats = ml_price.merge(ml_prob, on="merge_key", how="outer")
        ml_stats["ml_away_consensus"] = np.nan
    else:
        ml_stats = pd.DataFrame(columns=["merge_key", "ml_home_consensus", "ml_away_consensus", "home_ml_prob_consensus"])

    # combine all consensus pieces
    out = spread_stats.merge(totals_stats, on="merge_key", how="outer")
    out = out.merge(ml_stats, on="merge_key", how="outer")

    logger.info("[market_ensemble] Wide aggregation produced %d games.", out.merge_key.nunique())
    return out


# ---------------------------------------------------------------------
# Market aggregation dispatcher
# ---------------------------------------------------------------------

def aggregate_market_from_odds(odds_df: pd.DataFrame, snapshot_type: str = "close") -> pd.DataFrame:
    """Aggregate odds snapshot DataFrame into market consensus features.

    This function chooses the appropriate aggregation routine based on the
    presence of normalized or wide columns. It filters to the desired
    snapshot type and normalizes merge_key casing.
    """
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=[
            "merge_key", "home_spread_consensus", "home_spread_dispersion",
            "total_consensus", "home_ml_prob_consensus",
            "ml_home_consensus", "ml_away_consensus"
        ])

    df = odds_df.copy()
    # filter by snapshot type
    if "snapshot_type" in df.columns:
        df = df[df.snapshot_type.str.lower() == snapshot_type.lower()]

    # normalize merge_key
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()

    # decide aggregation based on columns present
    if {"market", "side", "point"} <= set(df.columns):
        return _aggregate_from_normalized(df)

    return _aggregate_from_wide(df)


# ---------------------------------------------------------------------
# CORE ENSEMBLE
# ---------------------------------------------------------------------

def apply_market_ensemble(
    preds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    snapshot_type: str = "close",
    spread_to_prob_slope: float = -0.165,
    min_market_weight: float = 0.20,
    max_market_weight: float = 0.80,
) -> pd.DataFrame:
    """Blend model predictions with market consensus.

    This function takes a DataFrame of model predictions (preds_df) and a
    DataFrame of market odds (odds_df), aggregates the market data, and
    produces blended probabilities and fair lines. The merge_key is
    constructed on the fly if missing in preds_df. Market weights are
    determined by the dispersion of consensus spreads.
    """
    preds = preds_df.copy()

    # construct merge_key if missing
    if "merge_key" not in preds.columns:
        preds["merge_key"] = (
            preds.home_team.str.lower().str.strip() + "__" +
            preds.away_team.str.lower().str.strip() + "__" +
            preds.game_date.astype(str).str[:10]
        )

    preds["merge_key"] = preds["merge_key"].str.lower().str.strip()

    market = aggregate_market_from_odds(odds_df, snapshot_type=snapshot_type)
    merged = preds.merge(market, on="merge_key", how="left")

    # rename model outputs for clarity before blending
    merged["home_win_prob_model"] = merged["home_win_prob"]
    merged["fair_spread_model"] = merged["fair_spread"]
    merged["fair_total_model"] = merged["fair_total"]

    # compute market-only win probability: average of spread-implied and ML-implied
    merged["home_win_prob_market"] = merged.apply(
        lambda r:
            0.5 * spread_to_win_prob(r.home_spread_consensus, spread_to_prob_slope)
            + 0.5 * r.home_ml_prob_consensus
            if pd.notna(r.home_spread_consensus) and pd.notna(r.home_ml_prob_consensus)
            else spread_to_win_prob(r.home_spread_consensus, spread_to_prob_slope)
            if pd.notna(r.home_spread_consensus)
            else r.home_ml_prob_consensus,
        axis=1
    )

    # compute market weight based on spread dispersion
    merged["market_weight"] = merged.home_spread_dispersion.apply(
        lambda d: compute_dispersion_weight(
            d,
            min_weight=min_market_weight,
            max_weight=max_market_weight
        )
    )

    # blend model and market win probabilities in logit space
    merged["home_win_prob"] = merged.apply(
        lambda r:
            _sigmoid(
                (1 - r.market_weight) * _logit(r.home_win_prob_model)
                + r.market_weight * _logit(r.home_win_prob_market)
            )
            if pd.notna(r.home_win_prob_market)
            else r.home_win_prob_model,
        axis=1
    )

    merged["away_win_prob"] = 1 - merged["home_win_prob"]

    # assign market fair lines
    merged["fair_spread_market"] = merged.home_spread_consensus
    merged["fair_total_market"] = merged.total_consensus

    # blended fair lines: weighted average of model and market
    merged["fair_spread"] = merged.apply(
        lambda r:
            (1 - r.market_weight) * r.fair_spread_model + r.market_weight * r.fair_spread_market
            if pd.notna(r.fair_spread_market)
            else r.fair_spread_model,
        axis=1
    )

    merged["fair_total"] = merged.apply(
        lambda r:
            (1 - r.market_weight) * r.fair_total_model + r.market_weight * r.fair_total_market
            if pd.notna(r.fair_total_market)
            else r.fair_total_model,
        axis=1
    )

    # Backward compatibility aliases
    merged["consensus_close"] = merged.home_spread_consensus
    merged["book_dispersion"] = merged.home_spread_dispersion
    merged["consensus_total"] = merged.total_consensus

    return merged


# ---------------------------------------------------------------------
# CSV wrapper
# ---------------------------------------------------------------------

def apply_market_ensemble_from_csv(preds_csv: str, odds_csv: str) -> pd.DataFrame:
    """Convenience wrapper to apply market ensemble from CSV file paths."""
    return apply_market_ensemble(pd.read_csv(preds_csv), pd.read_csv(odds_csv))
