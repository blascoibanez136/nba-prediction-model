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

MONEYLINE ODDS CONTRACT (CRITICAL)
----------------------------------
ml_home_consensus and ml_away_consensus MUST be American odds only.

Rules:
- odds must be numeric and finite
- odds must be non-zero
- odds must satisfy abs(odds) >= 100
- DO NOT convert decimal odds -> American
- invalid odds become NaN upstream (before modeling/backtest/ROI)
- fail loudly if any 0 < abs(odds) < 100 survives after sanitization
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_ENSEMBLE_VERSION = "market_ensemble_ml_contract_v1_2025-12-14"


# ---------------------------------------------------------------------
# Basic math utilities
# ---------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------
# Moneyline odds sanitizer (American-only contract)
# ---------------------------------------------------------------------

def clean_american_ml(x) -> Optional[float]:
    """
    Enforce American-only ML odds contract.
    - Returns float odds if valid American (abs >= 100)
    - Returns None if invalid / missing / decimal-like / malformed
    """
    if x is None:
        return None
    try:
        o = float(x)
    except Exception:
        return None
    if o == 0 or math.isnan(o) or math.isinf(o):
        return None
    if abs(o) < 100:
        return None
    return o


def _sanitize_ml_columns(
    df: pd.DataFrame,
    home_col: str = "ml_home_consensus",
    away_col: str = "ml_away_consensus",
    *,
    context: str = "unknown",
) -> pd.DataFrame:
    """
    Apply American-only ML contract to consensus columns and emit instrumentation.
    """
    out = df.copy()
    if home_col not in out.columns and away_col not in out.columns:
        return out

    def _count_decimal_like(s: pd.Series) -> int:
        s_num = pd.to_numeric(s, errors="coerce")
        return int(((s_num.abs() < 100) & (s_num.abs() > 0)).sum())

    total_games = int(out["merge_key"].nunique()) if "merge_key" in out.columns else int(len(out))
    before_home_nonnull = int(out[home_col].notna().sum()) if home_col in out.columns else 0
    before_away_nonnull = int(out[away_col].notna().sum()) if away_col in out.columns else 0
    before_decimal_like_home = _count_decimal_like(out[home_col]) if home_col in out.columns else 0
    before_decimal_like_away = _count_decimal_like(out[away_col]) if away_col in out.columns else 0

    if home_col in out.columns:
        out[home_col] = out[home_col].apply(clean_american_ml)
    if away_col in out.columns:
        out[away_col] = out[away_col].apply(clean_american_ml)

    after_home_nonnull = int(out[home_col].notna().sum()) if home_col in out.columns else 0
    after_away_nonnull = int(out[away_col].notna().sum()) if away_col in out.columns else 0

    dropped_home = before_home_nonnull - after_home_nonnull
    dropped_away = before_away_nonnull - after_away_nonnull

    logger.info(
        "[market_ensemble] [%s] ML contract sanitize: games=%d home_nonnull %d->%d (dropped=%d; decimal_like_before=%d) "
        "away_nonnull %d->%d (dropped=%d; decimal_like_before=%d)",
        context,
        total_games,
        before_home_nonnull,
        after_home_nonnull,
        dropped_home,
        before_decimal_like_home,
        before_away_nonnull,
        after_away_nonnull,
        dropped_away,
        before_decimal_like_away,
    )

    def _survivors(col: str) -> pd.Series:
        s = pd.to_numeric(out[col], errors="coerce")
        return (s.abs() < 100) & (s.abs() > 0)

    survivors_home = _survivors(home_col).sum() if home_col in out.columns else 0
    survivors_away = _survivors(away_col).sum() if away_col in out.columns else 0
    survivors = int(survivors_home + survivors_away)

    if survivors > 0:
        cols = ["merge_key"]
        if home_col in out.columns:
            cols.append(home_col)
        if away_col in out.columns:
            cols.append(away_col)
        sample = out.loc[
            (_survivors(home_col) if home_col in out.columns else False)
            | (_survivors(away_col) if away_col in out.columns else False),
            cols,
        ].head(10)

        raise RuntimeError(
            f"[market_ensemble] ML contract violation after sanitization in context={context}: "
            f"found {survivors} rows with 0 < abs(odds) < 100 in consensus columns. Sample:\n{sample.to_string(index=False)}"
        )

    return out


# ---------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------

def american_to_prob(odds: float) -> Optional[float]:
    """Convert American moneyline odds to implied probability."""
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0 or math.isnan(o) or math.isinf(o):
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


# ---------------------------------------------------------------------
# Spread <-> win probability
# ---------------------------------------------------------------------

def spread_to_win_prob(spread_home: float, slope: float = -0.165) -> Optional[float]:
    """Map home spread to win probability using logistic curve."""
    if spread_home is None:
        return None
    try:
        s = float(spread_home)
    except (TypeError, ValueError):
        return None
    if math.isnan(s):
        return None
    return _sigmoid(slope * s)


def win_prob_to_spread(p_home: float, slope: float = -0.165) -> Optional[float]:
    """
    Invert spread_to_win_prob.
    p = sigmoid(slope * spread)  =>  spread = logit(p) / slope
    With slope negative, p>0.5 yields negative spreads (home favored), which matches market convention.
    """
    if p_home is None:
        return None
    try:
        p = float(p_home)
    except (TypeError, ValueError):
        return None
    if math.isnan(p) or math.isinf(p) or not (0.0 < p < 1.0):
        return None
    if slope == 0:
        return None
    return _logit(p) / float(slope)


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
    """Map spread dispersion to a blending weight between model and market."""
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


def _detect_pred_points_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Best-effort detection of predicted home/away points columns."""
    home_candidates = [
        "pred_home_points_model",
        "pred_home_points",
        "home_points_pred",
        "home_score_pred",
        "pred_home_score",
        "home_pred_points",
        "model_home_points",
    ]
    away_candidates = [
        "pred_away_points_model",
        "pred_away_points",
        "away_points_pred",
        "away_score_pred",
        "pred_away_score",
        "away_pred_points",
        "model_away_points",
    ]
    home_col = next((c for c in home_candidates if c in df.columns), None)
    away_col = next((c for c in away_candidates if c in df.columns), None)
    return home_col, away_col


def _compute_fair_spread_model_from_points(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Compute fair_spread_model from predicted points.

    Convention:
    predicted_margin = pred_home_pts - pred_away_pts
    fair_spread_model (home line) = -predicted_margin
    """
    hcol, acol = _detect_pred_points_cols(df)
    if not hcol or not acol:
        return None

    h = pd.to_numeric(df[hcol], errors="coerce")
    a = pd.to_numeric(df[acol], errors="coerce")

    coverage = float((h.notna() & a.notna()).mean())
    if coverage < 0.95:
        logger.warning(
            "[market_ensemble] Pred points coverage too low to compute fair_spread_model safely: coverage=%.3f home_col=%s away_col=%s",
            coverage,
            hcol,
            acol,
        )
        return None

    margin = h - a
    spread = -margin

    max_abs = float(pd.to_numeric(spread, errors="coerce").abs().max())
    if math.isnan(max_abs) or max_abs > 40:
        raise RuntimeError(
            f"[market_ensemble] fair_spread_model out of bounds from points-derived computation: max_abs={max_abs}. "
            f"Check predicted points scale. Detected cols: home={hcol} away={acol}"
        )

    logger.info(
        "[market_ensemble] Computed fair_spread_model from predicted points: home=%s away=%s (coverage=%.3f, max_abs=%.2f)",
        hcol,
        acol,
        coverage,
        max_abs,
    )
    return spread


def _compute_fair_spread_model_from_prob(df: pd.DataFrame, *, slope: float) -> Optional[pd.Series]:
    """
    Compute fair_spread_model from model win probability using inverse logistic.

    This is the correct Pro-Lite fallback when predicted points are not available.
    """
    prob_col = None
    for c in ["home_win_prob_model", "home_win_prob"]:
        if c in df.columns:
            prob_col = c
            break
    if not prob_col:
        return None

    p = pd.to_numeric(df[prob_col], errors="coerce")
    coverage = float(p.notna().mean())
    if coverage < 0.95:
        logger.warning(
            "[market_ensemble] Model prob coverage too low to compute fair_spread_model from probability: coverage=%.3f prob_col=%s",
            coverage,
            prob_col,
        )
        return None

    spread = p.apply(lambda x: win_prob_to_spread(x, slope=slope))
    spread = pd.to_numeric(spread, errors="coerce")

    max_abs = float(spread.abs().max())
    if math.isnan(max_abs) or max_abs > 40:
        raise RuntimeError(
            f"[market_ensemble] fair_spread_model out of bounds from prob-derived computation: max_abs={max_abs}. "
            f"Check prob calibration/scale and slope={slope} prob_col={prob_col}"
        )

    logger.info(
        "[market_ensemble] Computed fair_spread_model from model win prob: prob_col=%s (coverage=%.3f, max_abs=%.2f, slope=%.3f)",
        prob_col,
        coverage,
        max_abs,
        slope,
    )
    return spread


# ---------------------------------------------------------------------
# Normalized odds aggregation
# ---------------------------------------------------------------------

def _aggregate_from_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized odds into consensus statistics."""
    df = df.copy()
    df["market"] = df["market"].astype(str).str.lower()
    df["side"] = df["side"].astype(str).str.lower()
    df["point"] = pd.to_numeric(df.get("point"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")

    spreads = df[(df.market == "spreads") & (df.side == "home")]
    totals = df[(df.market == "totals") & (df.side == "over")]
    h2h = df[(df.market == "h2h") & (df.side.isin(["home", "away"]))]

    if not spreads.empty:
        spread_stats = (
            spreads.groupby("merge_key")["point"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
            .reset_index()
        )
    else:
        spread_stats = pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    if not totals.empty:
        totals_stats = (
            totals.groupby("merge_key")["point"]
            .mean()
            .rename("total_consensus")
            .reset_index()
        )
    else:
        totals_stats = pd.DataFrame(columns=["merge_key", "total_consensus"])

    if not h2h.empty:
        h2h = h2h.copy()
        h2h["price"] = h2h["price"].apply(clean_american_ml)
        h2h["prob"] = h2h["price"].apply(american_to_prob)

        ml_price = (
            h2h.pivot_table(index="merge_key", columns="side", values="price", aggfunc="mean")
            .rename(columns={"home": "ml_home_consensus", "away": "ml_away_consensus"})
            .reset_index()
        )

        ml_prob = (
            h2h.pivot_table(index="merge_key", columns="side", values="prob", aggfunc="mean")
            .rename(columns={"home": "home_ml_prob_raw", "away": "away_ml_prob_raw"})
            .reset_index()
        )

        ml_stats = ml_price.merge(ml_prob, on="merge_key", how="outer")

        def _devig_row(r: pd.Series) -> Optional[float]:
            ph = r.get("home_ml_prob_raw")
            pa = r.get("away_ml_prob_raw")
            if pd.notna(ph) and pd.notna(pa) and (ph + pa) > 0:
                return ph / (ph + pa)
            return ph

        ml_stats["home_ml_prob_consensus"] = ml_stats.apply(_devig_row, axis=1)
        ml_stats = ml_stats.drop(columns=[c for c in ["home_ml_prob_raw", "away_ml_prob_raw"] if c in ml_stats.columns])

    else:
        ml_stats = pd.DataFrame(columns=["merge_key", "ml_home_consensus", "ml_away_consensus", "home_ml_prob_consensus"])

    out = spread_stats.merge(totals_stats, on="merge_key", how="outer")
    out = out.merge(ml_stats, on="merge_key", how="outer")

    out = _sanitize_ml_columns(out, context="aggregate_from_normalized")
    return out


# ---------------------------------------------------------------------
# Wide odds aggregation (AUTO-DETECT columns)
# ---------------------------------------------------------------------

def _aggregate_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wide snapshot odds into consensus statistics."""
    df = df.copy()

    def _find(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    spread_col = _find(["spread_home_point", "spread_home", "spread"])
    total_col = _find(["total_point", "total"])
    ml_home_col = _find(["ml_home", "home_ml"])
    ml_away_col = _find(["ml_away", "away_ml"])

    df = _safe_to_numeric(df, [spread_col, total_col, ml_home_col, ml_away_col])

    if spread_col:
        spread_stats = (
            df.groupby("merge_key")[spread_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": "home_spread_consensus", "std": "home_spread_dispersion"})
            .reset_index()
        )
    else:
        spread_stats = pd.DataFrame(columns=["merge_key", "home_spread_consensus", "home_spread_dispersion"])

    if total_col:
        totals_stats = (
            df.groupby("merge_key")[total_col]
            .mean()
            .rename("total_consensus")
            .reset_index()
        )
    else:
        totals_stats = pd.DataFrame(columns=["merge_key", "total_consensus"])

    if ml_home_col and ml_away_col:
        ml = df[["merge_key", ml_home_col, ml_away_col]].copy()
        ml[ml_home_col] = ml[ml_home_col].apply(clean_american_ml)
        ml[ml_away_col] = ml[ml_away_col].apply(clean_american_ml)

        ml_price = (
            ml.groupby("merge_key")[[ml_home_col, ml_away_col]]
            .mean()
            .rename(columns={ml_home_col: "ml_home_consensus", ml_away_col: "ml_away_consensus"})
            .reset_index()
        )

        def _devig(r: pd.Series) -> Optional[float]:
            ph = american_to_prob(r[ml_home_col])
            pa = american_to_prob(r[ml_away_col])
            return ph / (ph + pa) if (ph is not None and pa is not None and (ph + pa) > 0) else ph

        ml["prob"] = ml.apply(_devig, axis=1)
        ml_prob = ml.groupby("merge_key")["prob"].mean().rename("home_ml_prob_consensus").reset_index()
        ml_stats = ml_price.merge(ml_prob, on="merge_key", how="outer")

    elif ml_home_col:
        ml = df[["merge_key", ml_home_col]].copy()
        ml[ml_home_col] = ml[ml_home_col].apply(clean_american_ml)

        ml_price = (
            ml.groupby("merge_key")[ml_home_col]
            .mean()
            .rename("ml_home_consensus")
            .reset_index()
        )

        ml["prob"] = ml[ml_home_col].apply(american_to_prob)
        ml_prob = ml.groupby("merge_key")["prob"].mean().rename("home_ml_prob_consensus").reset_index()
        ml_stats = ml_price.merge(ml_prob, on="merge_key", how="outer")
        ml_stats["ml_away_consensus"] = np.nan

    else:
        ml_stats = pd.DataFrame(columns=["merge_key", "ml_home_consensus", "ml_away_consensus", "home_ml_prob_consensus"])

    out = spread_stats.merge(totals_stats, on="merge_key", how="outer")
    out = out.merge(ml_stats, on="merge_key", how="outer")

    out = _sanitize_ml_columns(out, context="aggregate_from_wide")

    logger.info("[market_ensemble] Wide aggregation produced %d games.", out.merge_key.nunique() if "merge_key" in out.columns else len(out))
    return out


# ---------------------------------------------------------------------
# Market aggregation dispatcher
# ---------------------------------------------------------------------

def aggregate_market_from_odds(odds_df: pd.DataFrame, snapshot_type: str = "close") -> pd.DataFrame:
    """Aggregate odds snapshot DataFrame into market consensus features."""
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=[
            "merge_key", "home_spread_consensus", "home_spread_dispersion",
            "total_consensus", "home_ml_prob_consensus",
            "ml_home_consensus", "ml_away_consensus"
        ])

    df = odds_df.copy()
    if "snapshot_type" in df.columns:
        df = df[df.snapshot_type.astype(str).str.lower() == snapshot_type.lower()]

    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()

    logger.info(
        "[market_ensemble] aggregate_market_from_odds version=%s snapshot_type=%s rows=%d games=%d",
        MARKET_ENSEMBLE_VERSION,
        snapshot_type,
        len(df),
        df["merge_key"].nunique() if "merge_key" in df.columns else -1,
    )

    if {"market", "side", "point"} <= set(df.columns):
        out = _aggregate_from_normalized(df)
    else:
        out = _aggregate_from_wide(df)

    out = _sanitize_ml_columns(out, context="aggregate_market_from_odds_final_guard")
    return out


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
    """Blend model predictions with market consensus."""
    preds = preds_df.copy()

    if "merge_key" not in preds.columns:
        preds["merge_key"] = (
            preds.home_team.str.lower().str.strip() + "__" +
            preds.away_team.str.lower().str.strip() + "__" +
            preds.game_date.astype(str).str[:10]
        )

    preds["merge_key"] = preds["merge_key"].astype(str).str.lower().str.strip()

    market = aggregate_market_from_odds(odds_df, snapshot_type=snapshot_type)
    merged = preds.merge(market, on="merge_key", how="left")

    # rename model outputs for clarity before blending
    if "home_win_prob" in merged.columns:
        merged["home_win_prob_model"] = merged["home_win_prob"]
    if "fair_total" in merged.columns:
        merged["fair_total_model"] = merged["fair_total"]

    # ---- FIX: COMPUTE fair_spread_model ROBUSTLY ----
    # Preference order:
    #   1) predicted points (best)
    #   2) inverse-logistic from model win prob (Pro-Lite correct fallback)
    #   3) fallback to fair_spread only if not constant
    spread_from_points = _compute_fair_spread_model_from_points(merged)
    if spread_from_points is not None:
        merged["fair_spread_model"] = spread_from_points
    else:
        spread_from_prob = _compute_fair_spread_model_from_prob(merged, slope=spread_to_prob_slope)
        if spread_from_prob is not None:
            merged["fair_spread_model"] = spread_from_prob
        else:
            if "fair_spread" in merged.columns:
                merged["fair_spread_model"] = pd.to_numeric(merged["fair_spread"], errors="coerce")
                nun = int(merged["fair_spread_model"].nunique(dropna=True))
                if nun <= 1:
                    raise RuntimeError(
                        "[market_ensemble] fair_spread_model fallback source 'fair_spread' is constant or missing. "
                        "Cannot safely run ATS. Provide predicted points columns or ensure upstream fair_spread varies per game."
                    )
                logger.warning(
                    "[market_ensemble] Using fallback fair_spread -> fair_spread_model (no points/prob cols detected). nunique=%d",
                    nun,
                )
            else:
                raise RuntimeError(
                    "[market_ensemble] Cannot compute fair_spread_model: no predicted points columns, no prob columns, and no fair_spread present."
                )

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
    if "home_win_prob_model" in merged.columns:
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

    if "fair_total_model" in merged.columns:
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
