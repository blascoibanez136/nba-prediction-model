"""
Market-aware ensemble for NBA Pro-Lite.

Takes:
- model-based predictions (our side)
- market features (consensus close, dispersion, movement)

Returns:
- blended win prob
- blended fair spread
- blended fair total
"""

from __future__ import annotations
import math
from typing import Optional
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shrink_toward_market(
    df: pd.DataFrame,
    market_col: str = "consensus_close",
    model_col: str = "fair_spread",
    dispersion_col: str = "book_dispersion",
    min_weight: float = 0.15,
    max_weight: float = 0.85,
) -> pd.Series:
    """
    Blend model fair_spread with market consensus spread.

    Weight is a function of dispersion: lower dispersion -> higher market weight.
    """
    out = []
    for _, row in df.iterrows():
        market = row.get(market_col)
        model = row.get(model_col)

        if pd.isna(market) or pd.isna(model):
            out.append(model)
            continue

        disp = row.get(dispersion_col, 0.0)
        # map dispersion to [min_weight, max_weight]
        # small dispersion -> weight near max_weight
        # big dispersion -> weight near min_weight
        # simple smooth function
        inv = 1.0 / (1.0 + disp)
        w = min_weight + (max_weight - min_weight) * inv

        blended = w * market + (1.0 - w) * model
        out.append(blended)
    return pd.Series(out, index=df.index)


def apply_market_ensemble(
    predictions: pd.DataFrame,
    odds_dispersion: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    predictions: DataFrame with at least game_id, fair_spread, fair_total, home_win_prob
    odds_dispersion: DataFrame from Day 5 with game_id, book_dispersion, consensus_close
    """
    df = predictions.copy()

    if odds_dispersion is not None and not odds_dispersion.empty:
        df = df.merge(odds_dispersion, on="game_id", how="left")
    else:
        # add empty cols so code below doesn't break
        df["book_dispersion"] = None
        df["consensus_close"] = None

    # 1) shrink spreads
    df["fair_spread_market"] = shrink_toward_market(
        df,
        market_col="consensus_close",
        model_col="fair_spread",
        dispersion_col="book_dispersion",
    )

    # 2) win prob: simple soft pull toward 50% if market is tight
    # (for Pro-Lite we keep it simple)
    if "book_dispersion" in df.columns:
        tight = df["book_dispersion"].fillna(5.0)
        # lower dispersion -> stronger pull
        adj_wp = []
        for wp, disp in zip(df["home_win_prob"], tight):
            # calc weight like above
            inv = 1.0 / (1.0 + disp)
            w = 0.1 + 0.4 * inv  # between 0.1 and 0.5
            blended_wp = w * 0.5 + (1 - w) * wp
            adj_wp.append(blended_wp)
        df["home_win_prob_market"] = adj_wp
    else:
        df["home_win_prob_market"] = df["home_win_prob"]

    # 3) totals – we don’t have a market total yet in this flow, so keep as-is
    df["fair_total_market"] = df["fair_total"]

    return df
