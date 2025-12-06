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
        inv = 1.0 / (1.0 + disp)
        w = min_weight + (max_weight - min_weight) * inv

        blended = w * market + (1.0 - w) * model
        out.append(blended)

    return pd.Series(out, index=df.index)


def apply_market_ensemble(
    predictions: pd.DataFrame,
    odds_dispersion: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = predictions.copy()

    if odds_dispersion is not None and not odds_dispersion.empty:
        odds_disp = odds_dispersion.copy()

        # Decide what key to merge on
        key_col: Optional[str] = None

        # 1) Try game_id if present in both frames
        if "game_id" in df.columns:
            if "game_id" not in odds_disp.columns and "game_id" in odds_disp.index.names:
                odds_disp = odds_disp.reset_index("game_id")
            if "game_id" in odds_disp.columns:
                df["game_id"] = df["game_id"].astype(str)
                odds_disp["game_id"] = odds_disp["game_id"].astype(str)
                key_col = "game_id"

        # 2) Fall back to merge_key if available in both
        if key_col is None and "merge_key" in df.columns and "merge_key" in odds_disp.columns:
            df["merge_key"] = df["merge_key"].astype(str)
            odds_disp["merge_key"] = odds_disp["merge_key"].astype(str)
            key_col = "merge_key"

        # 3) Last resort: build merge_key from teams + date
        if key_col is None:
            needed = ["home_team", "away_team", "game_date"]
            if all(c in df.columns for c in needed) and all(c in odds_disp.columns for c in needed):
                for frame in (df, odds_disp):
                    frame["merge_key"] = (
                        frame["home_team"].astype(str)
                        + "_"
                        + frame["away_team"].astype(str)
                        + "_"
                        + frame["game_date"].astype(str)
                    )
                key_col = "merge_key"

        if key_col is None:
            raise KeyError(
                "Could not determine a key to merge predictions with odds. "
                f"Pred columns: {list(df.columns)} | Odds columns: {list(odds_disp.columns)}"
            )

        # Merge in only the needed market columns
        merge_cols = [col for col in ["book_dispersion", "consensus_close", key_col] if col in odds_disp.columns]
        df = df.merge(odds_disp[merge_cols], on=key_col, how="left")
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
        adj_wp = []
        for wp, disp in zip(df["home_win_prob"], tight):
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


if __name__ == "__main__":
    """
    CLI use:
    - expects outputs/predictions_<today>.csv (the one your pipeline already writes)
    - optionally uses outputs/odds_dispersion_latest.csv
    - writes outputs/predictions_<today>_market.csv
    """
    import os
    from datetime import date

    os.makedirs("outputs", exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    preds_path = f"outputs/predictions_{today}.csv"
    preds = pd.read_csv(preds_path)

    odds_path = "outputs/odds_dispersion_latest.csv"
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path)
    else:
        odds = None

    out = apply_market_ensemble(preds, odds)
    out_path = f"outputs/predictions_{today}_market.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ wrote {out_path}")
