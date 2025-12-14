"""
Enhanced moneyline ROI and edge bucket analysis with bet‑side support.

This module fixes the previous limitation where only home bets were considered.  It
computes vig‑free implied probabilities from moneylines, compares the model's raw
win probability to the market, and supports betting either the home or away
team based on whichever side has a sufficiently large positive edge.

Key features:

* **Separate calibration vs. selection:**  You may optionally pass a
  probability calibrator.  The calibrator is applied to the model's win
  probability and recorded for diagnostics, but the *raw* model probability
  (pre‑calibration) is always used to compute edges and make bet decisions.

* **Bet side logic:**  For each game, both the home and away edges are
  computed.  A bet is placed on the home team if its edge exceeds the
  threshold and is greater than or equal to the away edge.  A bet is placed on
  the away team if its edge exceeds the threshold and is greater than the
  home edge.  If neither side has a qualifying edge, no bet is made.

* **Side‑specific ROI:**  Summary metrics and ROI are reported separately for
  home bets, away bets, and the combined portfolio.  This allows you to
  verify that both sides behave reasonably and to detect any asymmetries.

The per‑bucket summaries remain available for the combined portfolio and
continue to group bets by edge magnitude.

Usage example:

    python -m src.eval.roi_analysis \
        --per_game outputs/backtest_per_game.csv \
        --edge 0.02 \
        --calibrator models/calibrator.joblib

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import pandas as pd

from src.model.calibration import load_calibrator, apply_calibrator


# -----------------------------------------------------------------------------
# Odds + payout helpers
# -----------------------------------------------------------------------------


def american_to_prob(odds: float) -> Optional[float]:
    """Implied probability from American odds (includes vig if using one side only)."""
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0 or math.isnan(o):
        return None
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)


def devig_two_way(p_home: Optional[float], p_away: Optional[float]) -> Optional[float]:
    """Return vig‑free home probability if both sides exist."""
    if p_home is None or p_away is None:
        return None
    if not (0 < p_home < 1 and 0 < p_away < 1):
        return None
    denom = p_home + p_away
    if denom <= 0:
        return None
    return p_home / denom


def payout_per_dollar(odds: float) -> Optional[float]:
    """
    Profit on a $1 stake (not including returned stake).
    +150 -> win profit = 1.50
    -120 -> win profit = 0.8333...
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0 or math.isnan(o):
        return None
    return (o / 100.0) if o > 0 else (100.0 / abs(o))


# -----------------------------------------------------------------------------
# Column detection
# -----------------------------------------------------------------------------


@dataclass
class Cols:
    merge_key: str
    date: str
    model_prob: str
    market_prob: Optional[str]
    home_win: str
    ml_home: str
    ml_away: Optional[str]


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_columns(df: pd.DataFrame) -> Cols:
    """Infer the key column names used by downstream logic."""
    merge_key = _pick_first(df, ["merge_key", "merge_key_norm"])
    date = _pick_first(df, ["game_date", "date", "commence_time"])
    home_win = _pick_first(df, ["home_win_actual", "home_win", "home_win_result"])

    # Probability columns: prefer blended "home_win_prob" if present.
    model_prob = _pick_first(df, ["home_win_prob", "home_win_prob_model", "home_prob_model"])
    market_prob = _pick_first(df, ["home_win_prob_market", "home_prob_market"])

    # Moneyline columns
    ml_home = _pick_first(df, ["ml_home_consensus", "ml_home", "home_ml", "home_ml_consensus"])
    ml_away = _pick_first(df, ["ml_away_consensus", "ml_away", "away_ml", "away_ml_consensus"])

    missing: List[str] = []
    for name, val in [
        ("merge_key", merge_key),
        ("date", date),
        ("model_prob", model_prob),
        ("home_win", home_win),
        ("ml_home", ml_home),
    ]:
        if not val:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}. Available columns: {sorted(df.columns.tolist())}"
        )
    return Cols(
        merge_key=merge_key,
        date=date,
        model_prob=model_prob,
        market_prob=market_prob,
        home_win=home_win,
        ml_home=ml_home,
        ml_away=ml_away,
    )


# -----------------------------------------------------------------------------
# Bet construction (with side selection)
# -----------------------------------------------------------------------------


def build_bets(
    df: pd.DataFrame, cols: Cols, edge_threshold: float, model_prob_col: str
) -> pd.DataFrame:
    """
    Build a betting DataFrame with side selection.

    Args:
        df: DataFrame containing per‑game predictions and market data.
        cols: Column metadata inferred by `detect_columns`.
        edge_threshold: Minimum edge (probability points) required to place a bet.
        model_prob_col: Name of the column holding the *raw* model home win probability.

    Returns:
        A copy of `df` with additional columns:
          - market_prob_home: vig‑free implied home probability
          - home_edge / away_edge: edges for each side
          - bet_side: "home", "away", or None
          - bet: bool flag indicating whether a bet was placed
          - payout_per_home / payout_per_away: profit per $1 for each side
          - profit: realized profit on a $1 stake (NaN for no bet)
          - stake: 1.0 for a placed bet, 0.0 otherwise
          - bet_win_bin: 1 if the bet won, 0 if lost, NaN if no bet
          - edge_used: the edge corresponding to the chosen side (home_edge or away_edge)
    """
    out = df.copy()

    # Ensure numeric types for probabilities and odds
    out[model_prob_col] = pd.to_numeric(out[model_prob_col], errors="coerce")
    out[cols.ml_home] = pd.to_numeric(out[cols.ml_home], errors="coerce")
    if cols.ml_away:
        out[cols.ml_away] = pd.to_numeric(out[cols.ml_away], errors="coerce")

    # Compute market implied probabilities
    p_home = out[cols.ml_home].apply(american_to_prob)
    p_away = out[cols.ml_away].apply(american_to_prob) if cols.ml_away else None

    if p_away is not None:
        out["market_prob_home"] = [
            devig_two_way(ph, pa) for ph, pa in zip(p_home.tolist(), p_away.tolist())
        ]
        out["market_prob_method"] = "devig_two_way"
    else:
        # Single side only (no away odds) → use implied prob directly
        out["market_prob_home"] = p_home
        out["market_prob_method"] = "single_side_implied"

    # Compute edges for each side
    out["home_edge"] = out[model_prob_col] - out["market_prob_home"]
    # For away: model away probability minus market away prob = -(home_edge)
    out["away_edge"] = -out["home_edge"]

    # Determine bet side: prefer side with highest edge if above threshold
    def pick_side(row) -> Optional[str]:
        he = row["home_edge"]
        ae = row["away_edge"]
        if pd.isna(he) or pd.isna(ae):
            return None
        # Both edges negative or below threshold → no bet
        if he < edge_threshold and ae < edge_threshold:
            return None
        # If both exceed threshold, choose the larger edge
        if he >= edge_threshold and he >= ae:
            return "home"
        if ae >= edge_threshold and ae > he:
            return "away"
        return None

    out["bet_side"] = out.apply(pick_side, axis=1)
    out["bet"] = out["bet_side"].notna()

    # Payouts for each side
    out["payout_per_home"] = out[cols.ml_home].apply(payout_per_dollar)
    if cols.ml_away:
        out["payout_per_away"] = out[cols.ml_away].apply(payout_per_dollar)
    else:
        out["payout_per_away"] = None

    # Normalize actual result to 0/1 for home win
    hw = out[cols.home_win]
    if hw.dtype == bool:
        out["home_win_bin"] = hw.astype(int)
    else:
        out["home_win_bin"] = pd.to_numeric(hw, errors="coerce")

    # Compute bet win flag and profit
    out["bet_win_bin"] = None
    out["profit"] = None
    out["stake"] = 0.0
    out["edge_used"] = None

    # Home bets
    mask_home = out["bet_side"] == "home"
    # Bet win if home actually won
    out.loc[mask_home, "bet_win_bin"] = out.loc[mask_home, "home_win_bin"]
    out.loc[mask_home, "edge_used"] = out.loc[mask_home, "home_edge"]
    # Profit: payout if win, -1 if loss
    out.loc[mask_home & (out["home_win_bin"] == 1), "profit"] = out.loc[
        mask_home & (out["home_win_bin"] == 1), "payout_per_home"
    ]
    out.loc[mask_home & (out["home_win_bin"] == 0), "profit"] = -1.0
    out.loc[mask_home, "stake"] = 1.0

    # Away bets
    mask_away = out["bet_side"] == "away"
    # Bet win if home lost (away won)
    out.loc[mask_away, "bet_win_bin"] = 1.0 - out.loc[mask_away, "home_win_bin"]
    out.loc[mask_away, "edge_used"] = out.loc[mask_away, "away_edge"]
    # Profit: payout per away if away won, else -1
    if cols.ml_away:
        out.loc[
            mask_away & (out["home_win_bin"] == 0), "profit"
        ] = out.loc[
            mask_away & (out["home_win_bin"] == 0), "payout_per_away"
        ]
        out.loc[mask_away & (out["home_win_bin"] == 1), "profit"] = -1.0
    else:
        # No away odds → cannot compute profit; set to NaN
        out.loc[mask_away, "profit"] = float("nan")
    out.loc[mask_away, "stake"] = 1.0

    return out


# -----------------------------------------------------------------------------
# Metrics summarization
# -----------------------------------------------------------------------------


def _summarize_portfolio(b: pd.DataFrame) -> Dict[str, float]:
    """Compute ROI summary for a subset of bets DataFrame."""
    total_stake = float(b["stake"].sum())
    total_profit = float(pd.to_numeric(b["profit"], errors="coerce").sum())
    roi = (total_profit / total_stake) if total_stake > 0 else float("nan")
    win_rate = float(b["bet_win_bin"].mean()) if len(b) else float("nan")
    return {
        "bets": int(len(b)),
        "stake": total_stake,
        "profit": total_profit,
        "roi": roi,
        "win_rate": win_rate,
        "avg_edge": float(b["edge_used"].mean()) if len(b) else float("nan"),
        "avg_model_prob": float(b["home_win_prob_model_used"].mean())
        if "home_win_prob_model_used" in b.columns and len(b)
        else float("nan"),
        "avg_market_prob": float(b["market_prob_home"].mean()) if len(b) else float("nan"),
    }


def summarize(bets_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Summarize betting results for home, away, and combined portfolios.

    Args:
        bets_df: DataFrame returned by `build_bets` containing bet flags and profit.

    Returns:
        metrics: nested dict with keys 'combined', 'home', 'away', each mapping to ROI
            summary for that portfolio.
        bucket: per‑edge bucket summary for combined bets.
    """
    bets_only = bets_df[bets_df["bet"]].copy()

    # Combined metrics
    combined_metrics = _summarize_portfolio(bets_only)

    # Home portfolio
    home_portfolio = bets_only[bets_only["bet_side"] == "home"].copy()
    home_metrics = _summarize_portfolio(home_portfolio)

    # Away portfolio
    away_portfolio = bets_only[bets_only["bet_side"] == "away"].copy()
    away_metrics = _summarize_portfolio(away_portfolio)

    metrics = {
        "combined": combined_metrics,
        "home": home_metrics,
        "away": away_metrics,
    }

    # Bucket summary for combined bets
    def bucketize(edge: pd.Series) -> pd.Series:
        # buckets in absolute probability points: 0–1%, 1–2%, 2–4%, 4–6%, 6%+
        bins = [-1e9, 0.01, 0.02, 0.04, 0.06, 1e9]
        labels = ["0–1%", "1–2%", "2–4%", "4–6%", "6%+"]
        return pd.cut(edge, bins=bins, labels=labels, include_lowest=True)

    bucket_df = bets_only.copy()
    bucket_df["bucket"] = bucketize(bucket_df["edge_used"].abs())
    bucket = (
        bucket_df.groupby("bucket", dropna=False)
        .agg(
            bets=("bet", "size"),
            win_rate=("bet_win_bin", "mean"),
            avg_edge=("edge_used", "mean"),
            avg_model_prob=("home_win_prob_model_used", "mean")
            if "home_win_prob_model_used" in bucket_df.columns
            else ("edge_used", "mean"),
            avg_market_prob=("market_prob_home", "mean"),
            roi=("profit", lambda s: float(pd.to_numeric(s, errors="coerce").sum()) / len(s) if len(s) else float("nan")),
        )
        .reset_index()
    )

    return metrics, bucket


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per_game",
        default="outputs/backtest_per_game.csv",
        help="CSV with per‑game predictions/results. Generated by backtest.py",
    )
    ap.add_argument(
        "--edge",
        type=float,
        default=0.02,
        help="Edge threshold in probability points (0.02 = 2%).  A bet is placed on the side whose edge exceeds this threshold and is the larger of the two.",
    )
    ap.add_argument(
        "--out_json",
        default="outputs/roi_metrics.json",
        help="Path to write summary metrics JSON (home/away/combined)",
    )
    ap.add_argument(
        "--out_bucket",
        default="outputs/roi_buckets.csv",
        help="Path to write per‑bucket summary CSV (combined)",
    )
    ap.add_argument(
        "--out_bets",
        default="outputs/roi_bets.csv",
        help="Path to write full bets CSV",
    )
    ap.add_argument(
        "--calibrator",
        default=None,
        help="Optional path to a calibrator (.joblib). When provided, the model probability column"
             " will be calibrated for diagnostic purposes (stored as home_win_prob_model_used) but not used for edge computation.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.per_game):
        raise FileNotFoundError(f"Missing {args.per_game}. Run backtest.py first.")

    df = pd.read_csv(args.per_game)
    cols = detect_columns(df)

    # Always preserve the raw model probability column for edge computation
    df = df.copy()
    df[cols.model_prob] = pd.to_numeric(df[cols.model_prob], errors="coerce")
    df["home_win_prob_model_raw"] = df[cols.model_prob]

    # Apply calibrator for diagnostics if provided
    if args.calibrator:
        calib = load_calibrator(args.calibrator)
        df["home_win_prob_model_calibrated"] = apply_calibrator(df[cols.model_prob], calib)
        df["home_win_prob_model_used"] = df["home_win_prob_model_calibrated"]
        print(
            f"[roi_analysis] Applied calibrator from {args.calibrator} to column '{cols.model_prob}'."
        )
    else:
        df["home_win_prob_model_used"] = df[cols.model_prob]

    # Build bets using raw model probability for edges
    bets = build_bets(df, cols, edge_threshold=args.edge, model_prob_col=cols.model_prob)
    metrics, bucket = summarize(bets)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    bucket.to_csv(args.out_bucket, index=False)
    bets.to_csv(args.out_bets, index=False)

    # Print combined summary to console for convenience
    combined = metrics["combined"]
    print(f"[roi] edge_threshold={args.edge:.4f}")
    print(
        f"[roi] bets={combined['bets']} stake={combined['stake']:.1f} profit={combined['profit']:.3f} roi={combined['roi']:.4f} win_rate={combined['win_rate']:.4f}"
    )
    print(f"[roi] wrote: {args.out_json}")
    print(f"[roi] wrote: {args.out_bucket}")
    print(f"[roi] wrote: {args.out_bets}")


if __name__ == "__main__":
    main()
