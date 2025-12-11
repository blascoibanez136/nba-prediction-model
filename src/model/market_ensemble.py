"""
Market-aware ensemble for NBA Pro-Lite / Elite.

This module is responsible for blending our model-based predictions with
market information from The Odds API.

It supports TWO odds formats:

1) NORMALIZED FORMAT (from src/ingest/odds_normalizer.py)
   Columns like:
       - merge_key
       - market ("spreads", "totals", "h2h")
       - side ("home", "away", "over", "under")
       - point (numeric line)
       - price (American odds)
       - snapshot_type ("open", "mid", "close")

2) WIDE SNAPSHOT FORMAT (from src/ingest/odds_snapshots.py)
   Columns like:
       - merge_key
       - book
       - ml_home, ml_away
       - spread_home_point, spread_home_price
       - spread_away_point, spread_away_price
       - total_point, total_over_price, total_under_price
       - snapshot_type

It provides:
    - apply_market_ensemble(preds_df, odds_df, ...)
        Core function: blends model and market at the DataFrame level.
    - apply_market_ensemble_from_csv(preds_csv_path, odds_csv_path, ...)
        Convenience wrapper for CSV inputs.
    - apply_market_adjustment(predictions_csv_path, odds_csv_path=None, ...)
        Backwards-compatible wrapper for the daily pipeline: reads CSVs,
        finds the latest close snapshot if needed, writes a *_market.csv file,
        and returns the output CSV path.

Outputs are designed to be "backtest-ready" with explicit columns for:
    - home_win_prob_model      (raw model)
    - home_win_prob_market     (derived from spread + moneyline)
    - home_win_prob            (blended)
    - fair_spread_model        (raw model)
    - fair_spread_market       (consensus market line)
    - fair_spread              (blended)
    - fair_total_model         (raw model)
    - fair_total_market        (consensus market total)
    - fair_total               (blended)
    - market_weight            (weight given to market in the blend)
    - home_spread_dispersion   (std dev of home spread across books)

For backwards compatibility with existing edge_picker code, we also expose:
    - consensus_close   (alias of home_spread_consensus)
    - book_dispersion   (alias of home_spread_dispersion)
"""

from __future__ import annotations

import glob
import logging
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic math utilities
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-6) -> float:
    """
    Safe logit transform with clipping.
    """
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _inv_logit(z: float) -> float:
    return _sigmoid(z)


def american_to_prob(odds: float) -> Optional[float]:
    """
    Convert American odds to implied probability (single-sided, no-vig).
    Returns None if odds is missing or invalid.
    """
    if odds is None:
        return None

    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None

    if o == 0.0 or math.isnan(o):
        return None

    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return -o / (-o + 100.0)


# ---------------------------------------------------------------------------
# Spread -> win-prob conversion
# ---------------------------------------------------------------------------

def spread_to_win_prob(
    spread_home: float,
    slope: float = -0.165,
) -> Optional[float]:
    """
    Convert a consensus home spread (points) to an approximate win probability.

    We use a simple logistic curve:

        logit(P(home wins)) = slope * spread_home

    with intercept 0 so that spread=0 -> 50%.

    The default slope (-0.165) is calibrated so that roughly:
        home -7  -> ~76% win prob
        home +7  -> ~24% win prob

    This is intentionally simple and can be tuned later using historical data.
    """
    if spread_home is None:
        return None

    try:
        s = float(spread_home)
    except (TypeError, ValueError):
        return None

    if math.isnan(s):
        return None

    z = slope * s
    return _inv_logit(z)


# ---------------------------------------------------------------------------
# Dispersion -> market weight
# ---------------------------------------------------------------------------

def compute_dispersion_weight(
    dispersion: Optional[float],
    *,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
    pivot: float = 1.5,
    sharpness: float = 2.0,
) -> float:
    """
    Map book dispersion (std dev of spreads across books) to [min_weight, max_weight].

        low dispersion  -> trust market more (weight closer to max_weight)
        high dispersion -> trust market less (weight closer to min_weight)

    We do this with a logistic curve centered at `pivot` dispersion.

    Parameters
    ----------
    dispersion : float or None
        Standard deviation of the home spreads across books.
    min_weight, max_weight : float
        Bounds for market weight.
    pivot : float
        Dispersion value where weight is ~ (min_weight + max_weight) / 2.
    sharpness : float
        Controls how quickly the weight changes around the pivot.

    Returns
    -------
    float
        Market weight in [min_weight, max_weight].
    """
    if dispersion is None:
        return 0.5

    try:
        d = float(dispersion)
    except (TypeError, ValueError):
        return 0.5

    if math.isnan(d):
        return 0.5

    # Lower dispersion => higher trust in market
    z = -sharpness * (d - pivot)
    base = _sigmoid(z)  # in (0, 1)
    return float(min_weight + base * (max_weight - max(min_weight, min(max_weight, 0)))) if False else \
        float(min_weight + base * (max_weight - min_weight))


# ---------------------------------------------------------------------------
# Odds aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_from_normalized(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate market data when odds_df is in normalized format with
    'market', 'side', 'point', and 'price' columns.
    """
    df = odds_df.copy()

    # Normalize text columns
    df["market"] = df["market"].astype(str).str.lower()
    df["side"] = df["side"].astype(str).str.lower()

    # Ensure numeric types
    if "point" in df.columns:
        df["point"] = pd.to_numeric(df["point"], errors="coerce")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # --- Spreads (home side) ---
    spreads = df[(df["market"] == "spreads") & (df["side"] == "home")]
    if not spreads.empty:
        spread_stats = (
            spreads.groupby("merge_key")["point"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
            .reset_index()
        )
    else:
        spread_stats = pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    # --- Totals (use Over line as the reference total) ---
    totals = df[(df["market"] == "totals") & (df["side"] == "over")]
    if not totals.empty:
        totals_stats = (
            totals.groupby("merge_key")["point"]
            .mean()
            .rename("total_consensus")
            .reset_index()
        )
    else:
        totals_stats = pd.DataFrame(columns=["merge_key", "total_consensus"])

    # --- Moneyline (home team) ---
    h2h = df[(df["market"] == "h2h") & (df["side"] == "home")]
    if not h2h.empty:
        h2h = h2h.copy()
        h2h["prob"] = h2h["price"].apply(american_to_prob)
        ml_stats = (
            h2h.groupby("merge_key")["prob"]
            .mean()
            .rename("home_ml_prob_consensus")
            .reset_index()
        )
    else:
        ml_stats = pd.DataFrame(columns=["merge_key", "home_ml_prob_consensus"])

    out = spread_stats.merge(totals_stats, on="merge_key", how="outer")
    out = out.merge(ml_stats, on="merge_key", how="outer")

    return out


def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert given columns to numeric with errors coerced to NaN.
    Handles strings like 'None', '', etc.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate_from_wide(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate market data when odds_df is in wide snapshot format with columns like:
        - spread_home_point
        - total_point
        - ml_home

    We compute:
        - home_spread_consensus  = mean(spread_home_point) per merge_key
        - home_spread_dispersion = std(spread_home_point) per merge_key
        - total_consensus        = mean(total_point) per merge_key
        - home_ml_prob_consensus = mean(american_to_prob(ml_home)) per merge_key
    """
    df = odds_df.copy()

    # Clean and cast numeric columns from strings ("None", "", etc.) to floats/NaN
    df = _safe_to_numeric(
        df,
        [
            "spread_home_point",
            "spread_away_point",
            "spread_home_price",
            "spread_away_price",
            "total_point",
            "total_over_price",
            "total_under_price",
            "ml_home",
            "ml_away",
        ],
    )

    # --- Spreads ---
    if "spread_home_point" in df.columns:
        spread_stats = (
            df.groupby("merge_key")["spread_home_point"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
            .reset_index()
        )
    else:
        spread_stats = pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    # --- Totals ---
    if "total_point" in df.columns:
        totals_stats = (
            df.groupby("merge_key")["total_point"]
            .mean()
            .rename("total_consensus")
            .reset_index()
        )
    else:
        totals_stats = pd.DataFrame(columns=["merge_key", "total_consensus"])

    # --- Moneyline (home) ---
    if "ml_home" in df.columns:
        ml_df = df[["merge_key", "ml_home"]].copy()
        ml_df["prob"] = ml_df["ml_home"].apply(american_to_prob)
        ml_stats = (
            ml_df.groupby("merge_key")["prob"]
            .mean()
            .rename("home_ml_prob_consensus")
            .reset_index()
        )
    else:
        ml_stats = pd.DataFrame(columns=["merge_key", "home_ml_prob_consensus"])

    out = spread_stats.merge(totals_stats, on="merge_key", how="outer")
    out = out.merge(ml_stats, on="merge_key", how="outer")

    # Log how many games actually have valid spread/total/ml data
    n_games = out["merge_key"].nunique()
    logger.info(
        "[market_ensemble] Wide aggregation produced %d games with market data.",
        n_games,
    )

    return out


def aggregate_market_from_odds(
    odds_df: pd.DataFrame,
    snapshot_type: str = "close",
) -> pd.DataFrame:
    """
    Aggregate market data from either normalized or wide snapshot formats.

    Expected normalized columns:
        merge_key, market, side, price, point, snapshot_type

    Expected wide columns:
        merge_key, book,
        ml_home, ml_away,
        spread_home_point, spread_home_price,
        spread_away_point, spread_away_price,
        total_point, total_over_price, total_under_price,
        snapshot_type

    Returns one row per merge_key with:
        - home_spread_consensus
        - home_spread_dispersion
        - total_consensus
        - home_ml_prob_consensus
    """
    if odds_df is None or odds_df.empty:
        logger.warning("[market_ensemble] odds_df is empty; no market aggregation will be applied.")
        return pd.DataFrame(
            columns=[
                "merge_key",
                "home_spread_consensus",
                "home_spread_dispersion",
                "total_consensus",
                "home_ml_prob_consensus",
            ]
        )

    df = odds_df.copy()

    # Filter to desired snapshot (usually "close"), if column exists
    if "snapshot_type" in df.columns:
        df = df[df["snapshot_type"].astype(str).str.lower() == snapshot_type.lower()]

    if df.empty:
        logger.warning(
            "[market_ensemble] No odds rows left after filtering for snapshot_type=%s.",
            snapshot_type,
        )
        return pd.DataFrame(
            columns=[
                "merge_key",
                "home_spread_consensus",
                "home_spread_dispersion",
                "total_consensus",
                "home_ml_prob_consensus",
            ]
        )

    cols = set(df.columns)

    # NORMALIZED FORMAT
    if {"market", "side", "point"} <= cols:
        logger.info("[market_ensemble] Aggregating market data from NORMALIZED format.")
        return _aggregate_from_normalized(df)

    # WIDE SNAPSHOT FORMAT
    if "spread_home_point" in cols or "total_point" in cols or "ml_home" in cols:
        logger.info("[market_ensemble] Aggregating market data from WIDE snapshot format.")
        return _aggregate_from_wide(df)

    logger.warning(
        "[market_ensemble] Odds DataFrame does not match expected normalized or wide formats; "
        "no market aggregation will be applied."
    )
    return pd.DataFrame(
        columns=[
            "merge_key",
            "home_spread_consensus",
            "home_spread_dispersion",
            "total_consensus",
            "home_ml_prob_consensus",
        ]
    )


# ---------------------------------------------------------------------------
# Core ensemble logic
# ---------------------------------------------------------------------------

def apply_market_ensemble(
    preds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    snapshot_type: str = "close",
    spread_to_prob_slope: float = -0.165,
    min_market_weight: float = 0.20,
    max_market_weight: float = 0.80,
) -> pd.DataFrame:
    """
    Blend model predictions with market information.

    Parameters
    ----------
    preds_df : DataFrame
        Output of model prediction step, plus merge_key.
        Must contain:
            - merge_key (preferred)
              or (home_team, away_team, game_date) so we can build merge_key.
            - home_win_prob
            - fair_spread
            - fair_total
    odds_df : DataFrame
        Odds data, either in normalized or wide snapshot format.
    snapshot_type : str, optional
        Which snapshot to use (default "close").
    spread_to_prob_slope : float, optional
        Slope parameter for spread -> win-prob logistic mapping.
    min_market_weight, max_market_weight : float, optional
        Bounds for market weight as a function of dispersion.

    Returns
    -------
    DataFrame
        preds_df with additional columns:
            - home_win_prob_model
            - home_win_prob_market
            - home_win_prob (overwritten with blended value)
            - away_win_prob
            - fair_spread_model
            - fair_spread_market
            - fair_spread (overwritten with blended value)
            - fair_total_model
            - fair_total_market
            - fair_total (overwritten with blended value)
            - market_weight
            - home_spread_dispersion
            - home_spread_consensus
            - total_consensus
            - consensus_close (alias of home_spread_consensus)
            - book_dispersion (alias of home_spread_dispersion)
    """
    if preds_df is None or preds_df.empty:
        logger.warning("[market_ensemble] preds_df is empty; returning unchanged.")
        return preds_df.copy()

    preds = preds_df.copy()

    # Ensure merge_key exists
    if "merge_key" not in preds.columns:
        required = {"home_team", "away_team", "game_date"}
        if not required <= set(preds.columns):
            raise ValueError(
                "preds_df must contain 'merge_key' or "
                "'home_team', 'away_team', 'game_date'."
            )
        preds["merge_key"] = (
            preds["home_team"].astype(str).str.strip().str.lower()
            + "__"
            + preds["away_team"].astype(str).str.strip().str.lower()
            + "__"
            + preds["game_date"].astype(str)
        )

    # Aggregate market features
    market = aggregate_market_from_odds(odds_df, snapshot_type=snapshot_type)

    merged = preds.merge(market, on="merge_key", how="left", suffixes=("", "_mkt"))

    # Preserve original model outputs
    if "home_win_prob" not in merged.columns:
        raise ValueError("preds_df must contain 'home_win_prob' column from the model.")
    if "fair_spread" not in merged.columns:
        raise ValueError("preds_df must contain 'fair_spread' column from the model.")
    if "fair_total" not in merged.columns:
        raise ValueError("preds_df must contain 'fair_total' column from the model.")

    merged["home_win_prob_model"] = merged["home_win_prob"]
    merged["fair_spread_model"] = merged["fair_spread"]
    merged["fair_total_model"] = merged["fair_total"]

    # --- Market side win probability ---
    # 1) Spread-derived win prob
    spread_probs = merged["home_spread_consensus"].apply(
        lambda s: spread_to_win_prob(s, slope=spread_to_prob_slope) if pd.notnull(s) else None
    )
    merged["spread_prob"] = spread_probs

    # 2) Moneyline-derived win prob from home_ml_prob_consensus (possibly NaN)
    # 3) Combine them
    def _choose_market_prob(row) -> Optional[float]:
        sp = row.get("spread_prob", None)
        ml = row.get("home_ml_prob_consensus", None)

        sp_is_num = isinstance(sp, (float, int)) and not math.isnan(float(sp))
        ml_is_num = isinstance(ml, (float, int)) and not math.isnan(float(ml))

        if sp_is_num and ml_is_num:
            return 0.5 * float(sp) + 0.5 * float(ml)
        if sp_is_num:
            return float(sp)
        if ml_is_num:
            return float(ml)
        return None

    merged["home_win_prob_market"] = merged.apply(_choose_market_prob, axis=1)

    # --- Compute market weights from dispersion ---
    merged["market_weight"] = merged["home_spread_dispersion"].apply(
        lambda d: compute_dispersion_weight(
            d,
            min_weight=min_market_weight,
            max_weight=max_market_weight,
        )
    )

    # --- Blend win probabilities in logit space ---
    blended_probs = []
    for _, row in merged.iterrows():
        p_model = row["home_win_prob_model"]
        p_market = row["home_win_prob_market"]
        w = row["market_weight"]

        try:
            p_model_f = float(p_model)
        except (TypeError, ValueError):
            blended_probs.append(p_market if p_market is not None else 0.5)
            continue

        # If no valid market prob, fall back to model only
        if p_market is None or (isinstance(p_market, float) and math.isnan(float(p_market))):
            blended_probs.append(p_model_f)
            continue

        try:
            z_model = _logit(p_model_f)
            z_market = _logit(float(p_market))
            z_blend = (1.0 - w) * z_model + w * z_market
            blended_probs.append(_inv_logit(z_blend))
        except Exception:
            # In case of weird numeric issues, keep model prob.
            blended_probs.append(p_model_f)

    merged["home_win_prob"] = blended_probs
    merged["away_win_prob"] = 1.0 - merged["home_win_prob"]

    # --- Blend spreads & totals linearly ---
    merged["fair_spread_market"] = merged["home_spread_consensus"]
    merged["fair_total_market"] = merged["total_consensus"]

    def _blend_line(model_val: float, market_val: float, w: float) -> float:
        if market_val is None or (isinstance(market_val, float) and math.isnan(market_val)):
            return model_val
        try:
            mv = float(market_val)
        except (TypeError, ValueError):
            return model_val
        if math.isnan(mv):
            return model_val
        return (1.0 - w) * float(model_val) + w * mv

    merged["fair_spread"] = [
        _blend_line(m, mk, w)
        for m, mk, w in zip(
            merged["fair_spread_model"],
            merged["fair_spread_market"],
            merged["market_weight"],
        )
    ]

    merged["fair_total"] = [
        _blend_line(m, mk, w)
        for m, mk, w in zip(
            merged["fair_total_model"],
            merged["fair_total_market"],
            merged["market_weight"],
        )
    ]

    # ------------------------------------------------------------------
    # Backwards-compatibility aliases for edge_picker.py, etc.
    # ------------------------------------------------------------------
    if "home_spread_consensus" in merged.columns and "consensus_close" not in merged.columns:
        merged["consensus_close"] = merged["home_spread_consensus"]

    if "home_spread_dispersion" in merged.columns and "book_dispersion" not in merged.columns:
        merged["book_dispersion"] = merged["home_spread_dispersion"]

    return merged


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def apply_market_ensemble_from_csv(
    preds_csv_path: str,
    odds_csv_path: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: load predictions & odds from CSV,
    apply the ensemble, and return the blended DataFrame.
    """
    preds = pd.read_csv(preds_csv_path)
    odds = pd.read_csv(odds_csv_path)
    out = apply_market_ensemble(preds, odds, **kwargs)
    return out


def _find_latest_snapshot_csv(
    snapshot_dir: str,
    snapshot_type: str = "close",
) -> Optional[str]:
    """
    Find the most recent snapshot CSV for the given snapshot_type.
    Files are expected to look like:
        {snapshot_type}_YYYYMMDD_HHMMSS.csv
    """
    pattern = os.path.join(snapshot_dir, f"{snapshot_type}_*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort()
    latest = candidates[-1]
    return latest


def apply_market_adjustment(
    predictions_csv_path: str,
    odds_csv_path: Optional[str] = None,
    snapshot_dir: str = "data/_snapshots",
    snapshot_type: str = "close",
    output_csv_path: Optional[str] = None,
    *_,  # absorb unused positional args for backwards compatibility
    **__,  # absorb unused keyword args for backwards compatibility
) -> str:
    """
    Backwards-compatible wrapper intended for use from run_daily.py.

    Behavior:
        - Load predictions CSV.
        - Load odds CSV:
            - If odds_csv_path is provided, use that.
            - Otherwise, find latest {snapshot_type}_*.csv in snapshot_dir.
        - If odds data is available, apply the market ensemble.
        - Write a new predictions CSV with blended outputs.
        - Return the output CSV path.

    Parameters
    ----------
    predictions_csv_path : str
        Path to the base model predictions CSV.
    odds_csv_path : str, optional
        Path to an odds CSV (normalized or wide snapshot). If None, we try
        to auto-discover the latest snapshot in snapshot_dir.
    snapshot_dir : str
        Directory containing snapshot CSVs.
    snapshot_type : str
        "open", "mid", or "close". Default is "close".
    output_csv_path : str, optional
        Where to write the blended predictions. If None, a *_market.csv
        file will be created next to predictions_csv_path.

    Returns
    -------
    str
        Path to the output (blended) predictions CSV.
    """
    logger.info(
        "[market_ensemble] Applying market adjustment to %s (snapshot_type=%s).",
        predictions_csv_path,
        snapshot_type,
    )

    preds = pd.read_csv(predictions_csv_path)

    if odds_csv_path is None:
        odds_csv_path = _find_latest_snapshot_csv(snapshot_dir, snapshot_type=snapshot_type)

    if odds_csv_path is None or not os.path.exists(odds_csv_path):
        logger.warning(
            "[market_ensemble] No odds CSV found (snapshot_type=%s). "
            "Proceeding with model-only predictions.",
            snapshot_type,
        )
        preds_out = preds.copy()

        # Add *_model copies and compatibility columns so edge_picker doesn't break
        if "home_win_prob" in preds_out.columns:
            preds_out["home_win_prob_model"] = preds_out["home_win_prob"]
        if "fair_spread" in preds_out.columns:
            preds_out["fair_spread_model"] = preds_out["fair_spread"]
        if "fair_total" in preds_out.columns:
            preds_out["fair_total_model"] = preds_out["fair_total"]

        if "consensus_close" not in preds_out.columns:
            preds_out["consensus_close"] = np.nan
        if "book_dispersion" not in preds_out.columns:
            preds_out["book_dispersion"] = np.nan

        if output_csv_path is None:
            base, ext = os.path.splitext(predictions_csv_path)
            output_csv_path = f"{base}_market{ext}"

        preds_out.to_csv(output_csv_path, index=False)
        return output_csv_path

    logger.info("[market_ensemble] Using odds from %s", odds_csv_path)
    odds = pd.read_csv(odds_csv_path)

    blended = apply_market_ensemble(
        preds,
        odds,
        snapshot_type=snapshot_type,
    )

    if output_csv_path is None:
        base, ext = os.path.splitext(predictions_csv_path)
        output_csv_path = f"{base}_market{ext}"

    blended.to_csv(output_csv_path, index=False)
    logger.info(
        "[market_ensemble] Wrote market-adjusted predictions to %s (%d rows).",
        output_csv_path,
        len(blended),
    )

    return output_csv_path
