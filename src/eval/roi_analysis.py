"""
Moneyline ROI and edge bucket analysis (market-anchored, bet-side correct).

This module computes ROI using REAL market moneyline prices (American odds),
not model-derived payouts. It supports betting either home or away based on
the maximum positive edge versus the vig-free market implied probability.

Key guarantees:
- Uses closing consensus ML odds: ml_home_consensus / ml_away_consensus
- Computes vig-free market probs by devig normalization
- Computes edge using RAW model home win prob (pre-calibration)
- Applies optional calibrator for diagnostics ONLY (does not drive selection)
- Settles profit using American odds and actual game result
- Writes roi_bets.csv with `result` and `odds_price` (non-negotiable for ROI)

CLI:
  python src/eval/roi_analysis.py --per_game outputs/backtest_per_game.csv --edge 0.02 --calibrator models/cal.joblib
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from src.model.calibration import load_calibrator, apply_calibrator


# -----------------------------------------------------------------------------
# Odds helpers
# -----------------------------------------------------------------------------

def american_to_implied_prob(odds: float) -> Optional[float]:
    """American odds -> implied probability (includes vig if only one side)."""
    try:
        o = float(odds)
    except Exception:
        return None
    if math.isnan(o) or o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    # o < 0
    return abs(o) / (abs(o) + 100.0)


def american_to_win_profit_per_unit(odds: float) -> Optional[float]:
    """
    Profit (not return) on a winning bet for 1 unit stake, given American odds.
      +150 => +1.50 units profit
      -120 => +0.8333 units profit
    """
    try:
        o = float(odds)
    except Exception:
        return None
    if math.isnan(o) or o == 0:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def devig_probs(home_odds: float, away_odds: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute vig-free probabilities by normalizing implied probs.
    """
    ph = american_to_implied_prob(home_odds)
    pa = american_to_implied_prob(away_odds)
    if ph is None or pa is None:
        return None, None
    s = ph + pa
    if s <= 0:
        return None, None
    return ph / s, pa / s


# -----------------------------------------------------------------------------
# Config + IO
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ROIConfig:
    per_game_path: str
    edge_threshold: float
    calibrator_path: Optional[str]
    out_dir: str = "outputs"


def _pick_raw_model_prob_col(df: pd.DataFrame) -> str:
    """
    Prefer explicit RAW model probability columns if present.
    Fallback to home_win_prob if that is all we have.
    """
    candidates = [
        "home_win_prob_model_raw",
        "home_win_prob_model",
        "home_win_prob",  # last resort
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[roi] No usable model prob column found. Tried: {candidates}")


def _ensure_result_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have home_win_actual (0/1) to settle results.
    """
    out = df.copy()
    if "home_win_actual" in out.columns:
        out["home_win_actual"] = pd.to_numeric(out["home_win_actual"], errors="coerce")
        return out

    # fallback: derive from scores
    if "home_score" in out.columns and "away_score" in out.columns:
        hs = pd.to_numeric(out["home_score"], errors="coerce")
        as_ = pd.to_numeric(out["away_score"], errors="coerce")
        out["home_win_actual"] = (hs > as_).astype(float)
        return out

    raise RuntimeError("[roi] Missing home_win_actual and cannot infer from scores.")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


# -----------------------------------------------------------------------------
# Core ROI
# -----------------------------------------------------------------------------

def build_bets(df: pd.DataFrame, edge_threshold: float, calibrator_path: Optional[str]) -> pd.DataFrame:
    """
    Build bet-level dataframe from per-game backtest file.
    """
    if df.empty:
        raise RuntimeError("[roi] per_game dataframe is empty")

    # Required market odds
    for c in ["ml_home_consensus", "ml_away_consensus"]:
        if c not in df.columns:
            raise RuntimeError(f"[roi] per_game missing required column: {c}")

    df = _ensure_result_cols(df)

    raw_prob_col = _pick_raw_model_prob_col(df)

    # optional calibrator: apply to df["home_win_prob"] (or raw_prob_col if home_win_prob not present)
    df = df.copy()
    calib_used_col = None
    if calibrator_path:
        cal = load_calibrator(calibrator_path)
        target_col = "home_win_prob" if "home_win_prob" in df.columns else raw_prob_col
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df[f"{target_col}_calibrated"] = apply_calibrator(cal, df[target_col])
        calib_used_col = f"{target_col}_calibrated"
        print(f"[roi_analysis] Applied calibrator from {calibrator_path} to column '{target_col}'.")

    # compute vig-free market probabilities
    home_odds = pd.to_numeric(df["ml_home_consensus"], errors="coerce")
    away_odds = pd.to_numeric(df["ml_away_consensus"], errors="coerce")

    market_home = []
    market_away = []
    for ho, ao in zip(home_odds.tolist(), away_odds.tolist()):
        ph, pa = devig_probs(ho, ao)
        market_home.append(ph)
        market_away.append(pa)

    df["market_prob_home"] = market_home
    df["market_prob_away"] = market_away

    # raw model prob (for edge + selection)
    df[raw_prob_col] = pd.to_numeric(df[raw_prob_col], errors="coerce")
    df["model_prob_home_raw"] = df[raw_prob_col]
    df["model_prob_away_raw"] = 1.0 - df["model_prob_home_raw"]

    # edges
    df["home_edge"] = df["model_prob_home_raw"] - df["market_prob_home"]
    df["away_edge"] = df["model_prob_away_raw"] - df["market_prob_away"]

    # choose bet side
    def choose_side(r) -> Tuple[bool, Optional[str], Optional[float]]:
        he = r["home_edge"]
        ae = r["away_edge"]
        if pd.isna(he) or pd.isna(ae):
            return False, None, None
        # pick best side if above threshold
        if he >= edge_threshold and he >= ae:
            return True, "home", float(he)
        if ae >= edge_threshold and ae > he:
            return True, "away", float(ae)
        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0].astype(bool)
    df["bet_side"] = chosen[1]
    df["edge_used"] = chosen[2]

    bets = df[df["bet"]].copy()
    if bets.empty:
        return bets

    # odds_price per bet (American odds)
    bets["odds_price"] = bets.apply(
        lambda r: r["ml_home_consensus"] if str(r["bet_side"]).lower() == "home" else r["ml_away_consensus"],
        axis=1,
    )
    bets["odds_price"] = pd.to_numeric(bets["odds_price"], errors="coerce")

    # settle result
    bets["home_win_actual"] = pd.to_numeric(bets["home_win_actual"], errors="coerce")
    bets["result"] = bets.apply(
        lambda r: "win"
        if (str(r["bet_side"]).lower() == "home" and r["home_win_actual"] == 1)
        or (str(r["bet_side"]).lower() == "away" and r["home_win_actual"] == 0)
        else "loss",
        axis=1,
    )

    # stake (flat 1u unless column exists and is numeric)
    if "stake" in bets.columns:
        bets["stake"] = pd.to_numeric(bets["stake"], errors="coerce").fillna(1.0)
    else:
        bets["stake"] = 1.0

    # profit in units
    def profit_units(r) -> float:
        stake = float(r["stake"])
        if str(r["result"]).lower() != "win":
            return -stake
        ppu = american_to_win_profit_per_unit(r["odds_price"])
        if ppu is None:
            # If odds missing, treat as 0 profit (but this should not happen if consensus odds present)
            return 0.0
        return float(ppu * stake)

    bets["profit"] = bets.apply(profit_units, axis=1)

    # include calibrated prob for diagnostics if present
    if calib_used_col and calib_used_col in bets.columns:
        bets["model_prob_home_calibrated"] = bets[calib_used_col]

    # ensure merge_key exists (your backtest file already has it)
    if "merge_key" not in bets.columns and {"home_team", "away_team", "game_date"} <= set(bets.columns):
        # fallback; but typically merge_key is present
        bets["merge_key"] = (
            bets["home_team"].astype(str).str.lower().str.strip()
            + "__"
            + bets["away_team"].astype(str).str.lower().str.strip()
            + "__"
            + bets["game_date"].astype(str).str[:10]
        )

    # keep output columns + preserve additional columns if useful
    core_cols = [
        "game_date",
        "home_team",
        "away_team",
        "merge_key",
        "bet_side",
        "odds_price",
        "result",
        "stake",
        "profit",
        "edge_used",
        "home_edge",
        "away_edge",
        "market_prob_home",
        "market_prob_away",
        "model_prob_home_raw",
        "model_prob_away_raw",
    ]
    extra = [c for c in core_cols if c in bets.columns]
    # Keep all columns, but move core to front
    bets = bets[extra + [c for c in bets.columns if c not in extra]]
    return bets


def summarize_roi(bets: pd.DataFrame) -> Dict[str, Any]:
    stake = float(bets["stake"].sum()) if not bets.empty else 0.0
    profit = float(bets["profit"].sum()) if not bets.empty else 0.0
    roi = (profit / stake) if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean()) if not bets.empty else None
    return {
        "bets": int(len(bets)),
        "stake": stake,
        "profit": profit,
        "roi": roi,
        "win_rate": win_rate,
    }


def bucket_summary(bets: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket bets by edge_used magnitude.
    """
    if bets.empty:
        return pd.DataFrame(columns=["bucket", "bets", "stake", "profit", "roi", "win_rate"])

    b = bets.copy()
    b["edge_used"] = pd.to_numeric(b["edge_used"], errors="coerce")

    # buckets: [0.02,0.03), [0.03,0.04), ... [0.10, inf)
    edges = b["edge_used"].dropna()
    if edges.empty:
        b["bucket"] = "unknown"
    else:
        # fixed bins up to 0.10 then overflow
        bins = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 10.0]
        labels = [
            "<0.02",
            "0.02-0.03",
            "0.03-0.04",
            "0.04-0.05",
            "0.05-0.06",
            "0.06-0.07",
            "0.07-0.08",
            "0.08-0.09",
            "0.09-0.10",
            ">=0.10",
        ]
        b["bucket"] = pd.cut(b["edge_used"].fillna(-1.0), bins=bins, labels=labels, right=False, include_lowest=True)

    g = b.groupby("bucket", dropna=False, observed=False)
    out = g.agg(
        bets=("profit", "size"),
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        win_rate=("result", lambda s: (s.astype(str).str.lower() == "win").mean()),
        avg_edge=("edge_used", "mean"),
        avg_odds=("odds_price", lambda s: pd.to_numeric(s, errors="coerce").mean()),
    ).reset_index()

    out["roi"] = out.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else None, axis=1)
    return out


def main() -> None:
    p = argparse.ArgumentParser("roi_analysis.py")
    p.add_argument("--per_game", required=True, help="Path to outputs/backtest_per_game.csv")
    p.add_argument("--edge", required=True, type=float, help="Edge threshold (e.g., 0.02)")
    p.add_argument("--calibrator", default=None, help="Optional joblib calibrator path")
    args = p.parse_args()

    cfg = ROIConfig(
        per_game_path=args.per_game,
        edge_threshold=float(args.edge),
        calibrator_path=args.calibrator if args.calibrator else None,
        out_dir="outputs",
    )

    df = pd.read_csv(cfg.per_game_path)
    bets = build_bets(df, cfg.edge_threshold, cfg.calibrator_path)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Summaries
    print(f"[roi] edge_threshold={cfg.edge_threshold:.4f}")

    overall = summarize_roi(bets)
    print(f"[roi] bets={overall['bets']} stake={overall['stake']:.1f} profit={overall['profit']:.3f} "
          f"roi={overall['roi']:.4f} win_rate={overall['win_rate']:.4f}" if overall["bets"] else
          "[roi] bets=0 stake=0 profit=0 roi=nan win_rate=nan")

    home_bets = bets[bets["bet_side"].astype(str).str.lower() == "home"].copy() if not bets.empty else bets
    away_bets = bets[bets["bet_side"].astype(str).str.lower() == "away"].copy() if not bets.empty else bets

    home_sum = summarize_roi(home_bets)
    away_sum = summarize_roi(away_bets)

    # bucket summary (combined)
    buckets = bucket_summary(bets)

    metrics = {
        "edge_threshold": cfg.edge_threshold,
        "overall": overall,
        "home_only": home_sum,
        "away_only": away_sum,
        "calibrator": cfg.calibrator_path,
        "notes": {
            "settlement": "American odds on consensus close moneyline",
            "market_prob": "devig normalization of implied probs from ml_home_consensus/ml_away_consensus",
            "edge_source": "raw model home win prob (pre-calibration)",
        },
    }

    metrics_path = os.path.join(cfg.out_dir, "roi_metrics.json")
    buckets_path = os.path.join(cfg.out_dir, "roi_buckets.csv")
    bets_path = os.path.join(cfg.out_dir, "roi_bets.csv")

    _write_json(metrics_path, metrics)
    buckets.to_csv(buckets_path, index=False)
    bets.to_csv(bets_path, index=False)

    print(f"[roi] wrote: {metrics_path}")
    print(f"[roi] wrote: {buckets_path}")
    print(f"[roi] wrote: {bets_path}")


if __name__ == "__main__":
    main()

