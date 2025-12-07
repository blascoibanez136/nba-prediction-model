"""
Market-aware ensemble for NBA Pro-Lite.

Takes:
- model-based predictions (our side)
- market features (consensus close, dispersion, movement) via odds dispersion file

Returns:
- blended win prob (home_win_prob_market)
- blended fair spread (fair_spread_market)
- blended fair total (fair_total_market; currently same as model)

Merge strategy:
- We join predictions with odds dispersion on `merge_key`
- If `merge_key` is missing, we build it from home_team, away_team, game_date
"""

from __future__ import annotations

import math
from typing import Optional, Any, Dict, List

import pandas as pd


# -----------------------
# helpers (normalization)
# -----------------------
def _norm_team(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        # extend as needed
    }
    return aliases.get(s, s)


def _merge_key(home_team: str, away_team: str, game_date: str) -> str:
    """
    Build a stable cross-provider key:
      norm(home_team) + '__' + norm(away_team) + '__' + game_date (YYYY-MM-DD)
    """
    return f"{_norm_team(home_team)}__{_norm_team(away_team)}__{game_date}"


def _ensure_merge_key(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Ensure a DataFrame has a merge_key column built from home_team/away_team/game_date
    when possible. Returns a copy with merge_key present (or unchanged if not possible).
    """
    df = df.copy()

    if "merge_key" in df.columns:
        return df

    needed = {"home_team", "away_team", "game_date"}
    if not needed.issubset(df.columns):
        print(
            f"[market_ensemble] Warning: {source_name} has no 'merge_key' and is missing "
            f"one of {needed}. Cannot construct merge_key; continuing without join."
        )
        return df

    df["merge_key"] = df.apply(
        lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
    )
    return df


# -----------------------
# market shrink logic
# -----------------------
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

    If market or model is missing for a row, we fall back to the model.
    """
    out: List[Optional[float]] = []

    for _, row in df.iterrows():
        market = row.get(market_col)
        model = row.get(model_col)

        if pd.isna(market) or pd.isna(model):
            out.append(model)
            continue

        disp = row.get(dispersion_col, 0.0)
        try:
            disp_val = float(disp)
        except Exception:
            disp_val = 0.0

        # map dispersion to [min_weight, max_weight]
        # small dispersion -> weight near max_weight
        # big dispersion -> weight near min_weight
        inv = 1.0 / (1.0 + max(disp_val, 0.0))
        w = min_weight + (max_weight - min_weight) * inv

        blended = w * float(market) + (1.0 - w) * float(model)
        out.append(blended)

    return pd.Series(out, index=df.index)


# -----------------------
# main ensemble function
# -----------------------
def apply_market_ensemble(
    predictions: pd.DataFrame,
    odds_dispersion: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply a simple market-aware ensemble to model predictions.

    Inputs:
        predictions:
            DataFrame with at least:
              - home_team
              - away_team
              - game_date
              - home_win_prob
              - fair_spread
              - fair_total
            It may or may not already contain:
              - merge_key
              - book_dispersion
              - consensus_close

        odds_dispersion (optional):
            DataFrame from outputs/odds_dispersion_latest.csv with columns:
              - merge_key (or home_team, away_team, game_date to reconstruct it)
              - consensus_close
              - book_dispersion
            If missing or empty, we fall back to model-only outputs.

    Returns:
        DataFrame with additional columns:
          - fair_spread_market
          - home_win_prob_market
          - fair_total_market
          - (consensus_close, book_dispersion if available)
    """
    df = predictions.copy()

    # Ensure merge_key on predictions
    df = _ensure_merge_key(df, "predictions")

    # Attach odds dispersion if present
    if odds_dispersion is not None and not odds_dispersion.empty:
        disp = odds_dispersion.copy()
        disp = _ensure_merge_key(disp, "odds_dispersion")

        required_cols = {"merge_key", "consensus_close", "book_dispersion"}
        if not required_cols.issubset(disp.columns):
            missing = required_cols - set(disp.columns)
            print(
                f"[market_ensemble] Warning: odds_dispersion missing {missing}; "
                "skipping market blend and returning model-only outputs."
            )
            # Make sure columns exist but are NaN so downstream code doesn't break
            for col in ["consensus_close", "book_dispersion"]:
                if col not in df.columns:
                    df[col] = pd.NA
        else:
            # Only keep the relevant columns to avoid duplicate join columns
            disp_small = disp[["merge_key", "consensus_close", "book_dispersion"]].copy()
            df = df.merge(disp_small, on="merge_key", how="left", suffixes=("", "_odds"))

            # If for some reason predictions already had these, prefer the odds version
            for col in ["consensus_close", "book_dispersion"]:
                odds_col = f"{col}_odds"
                if odds_col in df.columns:
                    df[col] = df[col].where(df[col].notna(), df[odds_col])
                    df.drop(columns=[odds_col], inplace=True)
    else:
        # No odds dispersion: ensure columns exist but are empty
        if "consensus_close" not in df.columns:
            df["consensus_close"] = pd.NA
        if "book_dispersion" not in df.columns:
            df["book_dispersion"] = pd.NA

    # 1) fair_spread_market: shrink toward market spread when available
    if "consensus_close" in df.columns and "fair_spread" in df.columns:
        df["fair_spread_market"] = shrink_toward_market(
            df,
            market_col="consensus_close",
            model_col="fair_spread",
            dispersion_col="book_dispersion",
        )
    else:
        # Fallback: no market info, just copy model fair_spread
        if "fair_spread" in df.columns:
            df["fair_spread_market"] = df["fair_spread"]
        else:
            print(
                "[market_ensemble] Warning: 'fair_spread' missing in predictions; "
                "cannot create fair_spread_market."
            )
            df["fair_spread_market"] = pd.NA

    # 2) win prob: soft pull toward 50% if market dispersion is tight
    if "home_win_prob" not in df.columns:
        raise KeyError(
            "[market_ensemble] Required column 'home_win_prob' not found in predictions."
        )

    tight = df["book_dispersion"]
    # Default to 5.0 when dispersion is missing or invalid
    tight = pd.to_numeric(tight, errors="coerce").fillna(5.0)

    adj_wp: List[float] = []
    for wp_raw, disp in zip(df["home_win_prob"], tight):
        try:
            wp = float(wp_raw)
        except Exception:
            wp = 0.5  # neutral if something is badly malformed

        # lower dispersion -> stronger pull toward 0.5
        inv = 1.0 / (1.0 + max(disp, 0.0))
        w = 0.1 + 0.4 * inv  # between 0.1 and 0.5
        blended_wp = w * 0.5 + (1.0 - w) * wp
        adj_wp.append(blended_wp)

    df["home_win_prob_market"] = adj_wp

    # 3) totals – no market totals yet, so keep as-is
    if "fair_total" in df.columns:
        df["fair_total_market"] = df["fair_total"]
    else:
        print(
            "[market_ensemble] Warning: 'fair_total' missing in predictions; "
            "cannot create fair_total_market."
        )
        df["fair_total_market"] = pd.NA

    return df


# -----------------------
# CLI entry point
# -----------------------
if __name__ == "__main__":
    """
    CLI use:
    - expects outputs/predictions_<today>.csv (your pipeline's base predictions)
    - optionally uses outputs/odds_dispersion_latest.csv
    - writes outputs/predictions_<today>_market.csv
    """
    import os
    from datetime import date

    os.makedirs("outputs", exist_ok=True)
    today = os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")

    preds_path = f"outputs/predictions_{today}.csv"
    if not os.path.exists(preds_path):
        raise FileNotFoundError(
            f"[market_ensemble] Base predictions file not found: {preds_path}"
        )

    preds = pd.read_csv(preds_path)

    odds_path = "outputs/odds_dispersion_latest.csv"
    if os.path.exists(odds_path):
        odds = pd.read_csv(odds_path)
    else:
        print(
            "[market_ensemble] No odds_dispersion_latest.csv found; "
            "running model-only ensemble (no market blend)."
        )
        odds = None

    out = apply_market_ensemble(preds, odds)
    out_path = f"outputs/predictions_{today}_market.csv"
    out.to_csv(out_path, index=False)
    print(f"[market_ensemble] ✅ wrote {out_path} ({len(out)} rows)")
