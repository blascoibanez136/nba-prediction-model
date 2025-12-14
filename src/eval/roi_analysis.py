"""
Moneyline ROI and edge bucket analysis (with optional calibration).

This module provides utilities to evaluate moneyline betting strategies on
historical NBA data.  It computes flat‑stake ROI, win rates, and edge
distributions, and supports applying a pre‑trained probability calibrator
using isotonic regression.  The primary use case is to analyze the output
from ``src/eval/backtest.py`` (per‑game predictions and results) and
simulate simple betting rules.

Usage example:

    python -m roi_analysis \
        --per_game outputs/backtest_per_game.csv \
        --edge 0.02 \
        --out_json outputs/roi_metrics.json \
        --calibrator models/calibrator.joblib

If ``--calibrator`` is provided, the script will load the specified
calibrator and apply it to the model probability column before computing
edges.  This is useful to correct for systematic over/underconfidence in
predicted win probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from src.model.calibration import load_calibrator, apply_calibrator


# ----------------------------
# Odds + payout helpers
# ----------------------------

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


# ----------------------------
# Column detection
# ----------------------------

@dataclass
class Cols:
    merge_key: str
    date: str
    model_prob: str
    market_prob: Optional[str]
    home_win: str
    ml_home: str
    ml_away: Optional[str]


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_columns(df: pd.DataFrame) -> Cols:
    merge_key = _pick_first(df, ["merge_key", "merge_key_norm"])
    date = _pick_first(df, ["game_date", "date", "commence_time"])
    home_win = _pick_first(df, ["home_win_actual", "home_win", "home_win_result"])

    # Prob columns: prefer blended "home_win_prob" (post market ensemble) if present
    model_prob = _pick_first(df, ["home_win_prob", "home_win_prob_model", "home_prob_model"])
    market_prob = _pick_first(df, ["home_win_prob_market", "home_prob_market"])

    # Moneyline columns
    ml_home = _pick_first(df, ["ml_home_consensus", "ml_home", "home_ml", "home_ml_consensus"])
    ml_away = _pick_first(df, ["ml_away_consensus", "ml_away", "away_ml", "away_ml_consensus"])

    missing = []
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


# ----------------------------
# Core analysis
# ----------------------------

def build_bets(df: pd.DataFrame, cols: Cols, edge_threshold: float) -> pd.DataFrame:
    out = df.copy()

    # Ensure numeric
    out[cols.model_prob] = pd.to_numeric(out[cols.model_prob], errors="coerce")
    out[cols.ml_home] = pd.to_numeric(out[cols.ml_home], errors="coerce")
    if cols.ml_away:
        out[cols.ml_away] = pd.to_numeric(out[cols.ml_away], errors="coerce")

    # Market implied home prob
    p_home = out[cols.ml_home].apply(american_to_prob)
    p_away = (
        out[cols.ml_away].apply(american_to_prob) if cols.ml_away else None
    )

    if p_away is not None:
        out["market_prob_home"] = [
            devig_two_way(ph, pa) for ph, pa in zip(p_home.tolist(), p_away.tolist())
        ]
        out["market_prob_method"] = "devig_two_way"
    else:
        out["market_prob_home"] = p_home
        out["market_prob_method"] = "single_side_implied"

    # Edge = model - market
    out["edge"] = out[cols.model_prob] - out["market_prob_home"]

    # Qualifying bets (HOME only, flat $1)
    out["bet"] = (
        (out["edge"] >= edge_threshold)
        & out["market_prob_home"].notna()
        & out[cols.model_prob].notna()
    )

    # Profit per $1 stake
    out["payout_per_1"] = out[cols.ml_home].apply(payout_per_dollar)

    # Normalize win column to 0/1
    hw = out[cols.home_win]
    if hw.dtype == bool:
        out["home_win_bin"] = hw.astype(int)
    else:
        out["home_win_bin"] = pd.to_numeric(hw, errors="coerce")

    # Profit: win -> +payout_per_1 ; lose -> -1
    out["profit"] = None
    out.loc[out["bet"] & (out["home_win_bin"] == 1), "profit"] = out.loc[
        out["bet"] & (out["home_win_bin"] == 1), "payout_per_1"
    ]
    out.loc[out["bet"] & (out["home_win_bin"] == 0), "profit"] = -1.0

    # Stakes (flat $1)
    out["stake"] = 0.0
    out.loc[out["bet"], "stake"] = 1.0

    return out


def bucketize(edge: pd.Series) -> pd.Series:
    # buckets in absolute probability points: 0–1%, 1–2%, 2–4%, 4–6%, 6%+
    bins = [-1e9, 0.01, 0.02, 0.04, 0.06, 1e9]
    labels = ["0–1%", "1–2%", "2–4%", "4–6%", "6%+"]
    return pd.cut(edge, bins=bins, labels=labels, include_lowest=True)


def summarize(bets_df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    b = bets_df[bets_df["bet"]].copy()

    total_stake = float(b["stake"].sum())
    total_profit = float(pd.to_numeric(b["profit"], errors="coerce").sum())
    roi = (total_profit / total_stake) if total_stake > 0 else float("nan")

    win_rate = (
        float((b["home_win_bin"] == 1).mean()) if len(b) else float("nan")
    )

    metrics = {
        "bets": int(len(b)),
        "stake": total_stake,
        "profit": total_profit,
        "roi": roi,
        "win_rate": win_rate,
        "avg_edge": float(b["edge"].mean()) if len(b) else float("nan"),
        "avg_model_prob": float(b["home_win_prob_model_used"].mean())
        if "home_win_prob_model_used" in b.columns and len(b)
        else float("nan"),
        "avg_market_prob": float(b["market_prob_home"].mean()) if len(b) else float("nan"),
        "market_prob_method": b["market_prob_method"].iloc[0]
        if len(b)
        else None,
    }

    # Bucket summary
    b["bucket"] = bucketize(b["edge"])
    bucket = (
        b.groupby("bucket", dropna=False)
        .agg(
            bets=("bet", "size"),
            win_rate=("home_win_bin", "mean"),
            avg_edge=("edge", "mean"),
            avg_model_prob=("home_win_prob_model_used", "mean")
            if "home_win_prob_model_used" in b.columns
            else ("edge", "mean"),
            avg_market_prob=("market_prob_home", "mean"),
            roi=("profit", lambda s: float(pd.to_numeric(s, errors="coerce").sum()) / len(s) if len(s) else float("nan")),
        )
        .reset_index()
    )

    return metrics, bucket


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per_game",
        default="outputs/backtest_per_game.csv",
        help="CSV with per‑game predictions/results.  Generated by backtest.py",
    )
    ap.add_argument(
        "--edge",
        type=float,
        default=0.02,
        help="Edge threshold in probability points (0.02 = 2%)",
    )
    ap.add_argument(
        "--out_json",
        default="outputs/roi_metrics.json",
        help="Path to write summary metrics JSON",
    )
    ap.add_argument(
        "--out_bucket",
        default="outputs/roi_buckets.csv",
        help="Path to write per‑bucket summary CSV",
    )
    ap.add_argument(
        "--out_bets",
        default="outputs/roi_bets.csv",
        help="Path to write full bets CSV",
    )
    ap.add_argument(
        "--calibrator",
        default=None,
        help="Optional path to a calibrator (.joblib).  When provided, the model probability column"
             " will be transformed via the calibrator before edge computation.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.per_game):
        raise FileNotFoundError(f"Missing {args.per_game}. Run backtest.py first.")

    df = pd.read_csv(args.per_game)
    cols = detect_columns(df)

    # If calibrator provided, apply it to model probability column
    df = df.copy()
    if args.calibrator:
        calib = load_calibrator(args.calibrator)
        # Apply calibrator and overwrite the model_prob column for edge computation
        df[cols.model_prob] = apply_calibrator(
            pd.to_numeric(df[cols.model_prob], errors="coerce"), calib
        )
        # Record which probability is used
        df["home_win_prob_model_used"] = df[cols.model_prob]
        print(f"[roi_analysis] Applied calibrator from {args.calibrator} to column '{cols.model_prob}'.")
    else:
        # Without calibration, still coerce numeric and record
        df[cols.model_prob] = pd.to_numeric(df[cols.model_prob], errors="coerce")
        df["home_win_prob_model_used"] = df[cols.model_prob]

    bets = build_bets(df, cols, edge_threshold=args.edge)
    metrics, bucket = summarize(bets)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    bucket.to_csv(args.out_bucket, index=False)
    bets.to_csv(args.out_bets, index=False)

    print(f"[roi] edge_threshold={args.edge:.4f}")
    print(
        f"[roi] bets={metrics['bets']} stake={metrics['stake']:.1f} profit={metrics['profit']:.3f} roi={metrics['roi']:.4f} win_rate={metrics['win_rate']:.4f}"
    )
    print(f"[roi] wrote: {args.out_json}")
    print(f"[roi] wrote: {args.out_bucket}")
    print(f"[roi] wrote: {args.out_bets}")


if __name__ == "__main__":
    main()
