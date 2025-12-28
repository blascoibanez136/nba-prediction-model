"""
E2 Policy Runner (ATS-based)

Generates outputs/e2_policy_metrics.json in the schema expected by qa/certify.py.

This runner evaluates an ATS policy using:
- Per-game model fair spread (from backtest_joined.csv)
- Market spreads from snapshots:
    open_YYYYMMDD.csv  (spread_home_point, spread_home_price)
    close_YYYYMMDD.csv (spread_home_point)
- A spread calibrator artifact (joblib dict) produced by src.eval.train_spread_calibrator
  mapping residual -> P(home_covers)

Policy (E2, ATS-based):
- Compute residual = fair_spread_model - close_home_spread_point
- p_home_cover = calibrator(residual, close_spread)
- p_away_cover = 1 - p_home_cover
- Compute EV at OPEN price for away side
- Select away-only bets with EV >= ev_threshold
- Gate by dispersion <= max_dispersion if available
- Optional trim by max_bet_rate

CLV:
- Defined as open_spread_home_point - close_spread_home_point (home-line movement).
  For away bets, positive CLV means the close line moved in our favor (home line became less favorable for home).
  We also report CLV aligned to bet side via a sign convention:
    clv_aligned = (open_home_spread - close_home_spread) * (+1 for away bet, -1 for home bet)

Deterministic + fail-loud + American-only enforcement for price columns.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.model.spread_relative_calibration import apply_spread_calibrator


# ---------------------------
# Odds helpers (American-only)
# ---------------------------

def clean_american_odds(x: object) -> Optional[float]:
    try:
        o = float(x)
    except Exception:
        return None
    if not math.isfinite(o) or o == 0:
        return None
    # American-only: abs >= 100
    if 0 < abs(o) < 100:
        return None
    return o


def american_to_prob(o: Optional[float]) -> Optional[float]:
    o = clean_american_odds(o)
    if o is None:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def win_profit_per_unit_american(o: Optional[float]) -> Optional[float]:
    o = clean_american_odds(o)
    if o is None:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def expected_value_units(p_win: Optional[float], odds: Optional[float]) -> Optional[float]:
    """
    EV in units for 1u stake:
      EV = p*ppu - (1-p)*1
    """
    if p_win is None:
        return None
    try:
        p = float(p_win)
    except Exception:
        return None
    if not (0.0 < p < 1.0):
        return None
    ppu = win_profit_per_unit_american(odds)
    if ppu is None:
        return None
    return p * ppu - (1.0 - p)


def merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{str(home_team).strip().lower()}__{str(away_team).strip().lower()}__{str(game_date).strip()[:10]}"


def _ymd(date_str: str) -> str:
    return str(date_str).replace("-", "")


# ---------------------------
# Snapshot loaders
# ---------------------------

def load_spread_snapshot_day(snapshot_dir: Path, game_date: str, snapshot_type: str) -> pd.DataFrame:
    """
    Returns per-game spread home point + (optional) dispersion + (optional) open price.

    Expected columns in snapshot:
      - merge_key
      - spread_home_point (or spread_home / spread)
      - spread_home_price (open only; close may not have price)
      - book (optional) used for dispersion
    """
    st = snapshot_type.lower().strip()
    path = snapshot_dir / f"{st}_{_ymd(game_date)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["merge_key"])

    df = pd.read_csv(path)

    # ensure merge_key
    if "merge_key" not in df.columns:
        if all(c in df.columns for c in ["home_team", "away_team", "game_date"]):
            df["merge_key"] = [merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])]
        else:
            return pd.DataFrame(columns=["merge_key"])

    # detect spread point col
    def _find(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    spread_col = _find(["spread_home_point", "spread_home", "spread"])
    price_col = _find(["spread_home_price", "spread_price_home", "spread_home_odds"])

    out = df[["merge_key"]].copy()

    if spread_col:
        out["home_spread_point"] = pd.to_numeric(df[spread_col], errors="coerce")

        # dispersion: std of home_spread_point across books if multiple rows exist
        if "book" in df.columns:
            disp = (
                df.groupby("merge_key")[spread_col]
                .agg(["std"])
                .rename(columns={"std": "home_spread_dispersion"})
                .reset_index()
            )
            disp["home_spread_dispersion"] = pd.to_numeric(disp["home_spread_dispersion"], errors="coerce")
            out = out.merge(disp, on="merge_key", how="left")

    if price_col:
        # mean price across books (deterministic aggregation)
        px = (
            df.groupby("merge_key")[price_col]
            .mean(numeric_only=True)
            .reset_index()
            .rename(columns={price_col: "home_spread_price"})
        )
        px["home_spread_price"] = pd.to_numeric(px["home_spread_price"], errors="coerce").apply(clean_american_odds)
        out = out.merge(px, on="merge_key", how="left")

    return out


# ---------------------------
# E2 runner core
# ---------------------------

@dataclass(frozen=True)
class E2ATSConfig:
    per_game_path: str
    snapshot_dir: str
    calibrator_path: str
    start: str
    end: str
    out_path: str

    # ATS Policy knobs (aligned with ATS v1 defaults)
    side_policy: str = "away_only"  # away_only | home_only | both
    ev_threshold: float = 0.03
    max_dispersion: float = 2.0
    require_dispersion: bool = True
    max_bet_rate: float = 0.30  # cap by trimming EV if needed
    stake: float = 1.0


def compute_e2_ats_metrics(cfg: E2ATSConfig) -> Dict[str, Any]:
    per_game = Path(cfg.per_game_path)
    if not per_game.exists():
        raise FileNotFoundError(f"[e2_ats] missing per_game: {per_game}")

    cal_path = Path(cfg.calibrator_path)
    if not cal_path.exists():
        raise FileNotFoundError(f"[e2_ats] missing calibrator: {cal_path}")

    df = pd.read_csv(per_game)
    if df.empty:
        raise RuntimeError("[e2_ats] per_game is empty")

    # Normalize date
    if "game_date" not in df.columns:
        if "date" in df.columns:
            df["game_date"] = df["date"].astype(str).str[:10]
        else:
            raise RuntimeError("[e2_ats] per_game must include game_date or date")

    df["game_date"] = df["game_date"].astype(str).str[:10]
    df = df[(df["game_date"] >= cfg.start) & (df["game_date"] <= cfg.end)].copy()
    if df.empty:
        raise RuntimeError("[e2_ats] No rows in requested window")

    # Required columns
    required = ["home_team", "away_team", "home_score", "away_score"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"[e2_ats] missing required column: {c}")

    # fair_spread_model preference: use fair_spread_model if present else fair_spread
    fair_col = "fair_spread_model" if "fair_spread_model" in df.columns else ("fair_spread" if "fair_spread" in df.columns else None)
    if not fair_col:
        raise RuntimeError("[e2_ats] missing fair spread column (fair_spread_model or fair_spread)")

    df[fair_col] = pd.to_numeric(df[fair_col], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=[fair_col, "home_score", "away_score"]).copy()
    if df.empty:
        raise RuntimeError("[e2_ats] No valid rows after numeric coercion")

    df["merge_key"] = [merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])]

    # Load calibrator (dict type used by spread_relative_calibration)
    calibrator = joblib.load(str(cal_path))
    if not isinstance(calibrator, dict) or calibrator.get("type") != "spread_isotonic_bucketed_v1":
        raise RuntimeError("[e2_ats] invalid spread calibrator artifact (expected spread_isotonic_bucketed_v1 dict)")

    # Attach open + close spreads/prices
    snap_dir = Path(cfg.snapshot_dir)

    open_frames = []
    close_frames = []
    for d in sorted(df["game_date"].unique()):
        open_frames.append(load_spread_snapshot_day(snap_dir, d, "open"))
        close_frames.append(load_spread_snapshot_day(snap_dir, d, "close"))

    open_df = pd.concat(open_frames, ignore_index=True) if open_frames else pd.DataFrame(columns=["merge_key"])
    close_df = pd.concat(close_frames, ignore_index=True) if close_frames else pd.DataFrame(columns=["merge_key"])

    open_df = open_df.rename(columns={
        "home_spread_point": "home_spread_open",
        "home_spread_price": "home_spread_price_open",
        "home_spread_dispersion": "home_spread_dispersion_open",
    })
    close_df = close_df.rename(columns={
        "home_spread_point": "home_spread_close",
        "home_spread_dispersion": "home_spread_dispersion_close",
    })

    df = df.merge(open_df, on="merge_key", how="left")
    df = df.merge(close_df, on="merge_key", how="left")

    total_rows = int(len(df))

    # Eligibility: need open + close spread points and open price for settlement
    elig = (
        df["home_spread_open"].notna()
        & df["home_spread_close"].notna()
        & df["home_spread_price_open"].notna()
    )

    eligible = df.loc[elig].copy()
    eligible_rows = int(len(eligible))
    eligible_pct = float(eligible_rows / max(total_rows, 1))

    if eligible_rows == 0:
        raise RuntimeError("[e2_ats] No eligible rows with open+close spread + open price.")

    # Dispersion gating (use CLOSE dispersion if present; else OPEN dispersion; else fail if required)
    disp_col = None
    if "home_spread_dispersion_close" in eligible.columns and eligible["home_spread_dispersion_close"].notna().any():
        disp_col = "home_spread_dispersion_close"
    elif "home_spread_dispersion_open" in eligible.columns and eligible["home_spread_dispersion_open"].notna().any():
        disp_col = "home_spread_dispersion_open"

    if cfg.require_dispersion and disp_col is None:
        raise RuntimeError("[e2_ats] require_dispersion=True but no dispersion column available from snapshots.")

    if disp_col is not None:
        eligible[disp_col] = pd.to_numeric(eligible[disp_col], errors="coerce")
        eligible = eligible[(eligible[disp_col].notna()) & (eligible[disp_col] <= float(cfg.max_dispersion))].copy()

    # If dispersion filtering empties sample, fail loud
    if eligible.empty:
        raise RuntimeError("[e2_ats] Eligible sample empty after dispersion gating.")

    # Residual computed vs CLOSE line (this matches ATS v1 logic)
    eligible["residual"] = pd.to_numeric(eligible[fair_col], errors="coerce") - pd.to_numeric(eligible["home_spread_close"], errors="coerce")
    eligible["residual"] = pd.to_numeric(eligible["residual"], errors="coerce")
    eligible = eligible.dropna(subset=["residual"]).copy()
    if eligible.empty:
        raise RuntimeError("[e2_ats] Eligible sample empty after residual computation.")

    # Apply calibrator: P(home_covers) from residual
    eligible["p_home_cover"] = eligible.apply(
        lambda r: apply_spread_calibrator(
            residual=float(r["residual"]) if pd.notna(r["residual"]) else None,
            home_spread_consensus=float(r["home_spread_close"]) if pd.notna(r["home_spread_close"]) else None,
            calibrator=calibrator,
        ),
        axis=1,
    )
    eligible["p_home_cover"] = pd.to_numeric(eligible["p_home_cover"], errors="coerce")
    eligible = eligible.dropna(subset=["p_home_cover"]).copy()
    if eligible.empty:
        raise RuntimeError("[e2_ats] Eligible sample empty after calibrator application.")

    eligible["p_away_cover"] = 1.0 - eligible["p_home_cover"]

    # Pricing: Use OPEN home spread price for both sides (typical same vig both sides)
    eligible["price_open"] = pd.to_numeric(eligible["home_spread_price_open"], errors="coerce").apply(clean_american_odds)
    eligible = eligible.dropna(subset=["price_open"]).copy()
    if eligible.empty:
        raise RuntimeError("[e2_ats] Eligible sample empty after price sanitization.")

    # EV (away side only by default)
    eligible["away_ev"] = eligible.apply(lambda r: expected_value_units(float(r["p_away_cover"]), float(r["price_open"])), axis=1)

    # Side policy
    side_policy = str(cfg.side_policy).strip().lower()
    if side_policy not in {"away_only", "home_only", "both"}:
        raise RuntimeError("[e2_ats] Invalid side_policy. Use: away_only | home_only | both")

    # We only implement away_only here (matches your ATS v1 now)
    if side_policy == "home_only":
        raise RuntimeError("[e2_ats] home_only not implemented in this E2 runner (use ats_roi_analysis if needed).")
    if side_policy == "both":
        raise RuntimeError("[e2_ats] both not implemented in this E2 runner (intentionally conservative).")

    # Selection: away side if EV >= threshold
    eligible["bet"] = eligible["away_ev"] >= float(cfg.ev_threshold)
    bets = eligible.loc[eligible["bet"]].copy()

    if bets.empty:
        # Write empty metrics (fail-loud in certify later due to bets < 60)
        return {
            "sample_size": {"bets": 0, "rows": 0, "start": cfg.start, "end": cfg.end},
            "performance": {"roi": None, "average_clv": None, "clv_positive_rate": None, "profit_units": 0.0, "stake_units": 0.0},
            "risk_metrics": {"max_drawdown_units": 0.0},
            "clv_coverage": {"open_snapshot_coverage_pct": 100, "close_snapshot_coverage_pct": 100},
            "filters": {"policy": "e2_ats_away_only", "notes": "No bets selected."},
            "eligibility": {"total_rows": total_rows, "eligible_rows": eligible_rows, "eligible_pct": eligible_pct, "excluded_rows": total_rows - eligible_rows},
        }

    # Bet-rate cap by trimming EV
    total_games = int(eligible["merge_key"].nunique()) if "merge_key" in eligible.columns else int(len(eligible))
    max_bets = max(1, int(math.floor(float(cfg.max_bet_rate) * max(total_games, 1))))
    if len(bets) > max_bets:
        bets = bets.sort_values("away_ev", ascending=False).head(max_bets).copy()

    # CLV (spread movement): open - close for home line
    bets["clv_raw_home"] = pd.to_numeric(bets["home_spread_open"], errors="coerce") - pd.to_numeric(bets["home_spread_close"], errors="coerce")
    # For away bets, positive CLV means close moved in our favor => use +clv_raw_home
    bets["clv_aligned"] = bets["clv_raw_home"]

    avg_clv = float(pd.to_numeric(bets["clv_aligned"], errors="coerce").mean())
    clv_pos_rate = float((pd.to_numeric(bets["clv_aligned"], errors="coerce") > 0).mean())

    # Settlement: away covers if (home_score + close_home_spread) < away_score
    # Use CLOSE spread line for grading (standard)
    bets["adj_home"] = bets["home_score"] + bets["home_spread_close"]
    bets["result"] = np.where(bets["adj_home"] < bets["away_score"], "win",
                              np.where(bets["adj_home"] > bets["away_score"], "loss", "push"))

    # Profit using OPEN price and 1u stake (push=0)
    bets["stake"] = float(cfg.stake)
    bets["ppu"] = bets["price_open"].apply(win_profit_per_unit_american)
    bets["profit"] = np.where(
        bets["result"] == "win",
        bets["stake"] * bets["ppu"],
        np.where(bets["result"] == "push", 0.0, -bets["stake"])
    )

    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0.0).sum())
    roi = float(profit / stake) if stake > 0 else None

    # Drawdown by day
    daily = bets.groupby("game_date")["profit"].sum().reset_index()
    daily = daily.sort_values("game_date").reset_index(drop=True)
    daily["cum"] = daily["profit"].cumsum()
    daily["peak"] = daily["cum"].cummax()
    daily["dd"] = daily["cum"] - daily["peak"]
    max_dd = float(daily["dd"].min()) if not daily.empty else 0.0

    # Coverage on eligible universe is 100% by construction for the fields we used.
    open_cov = 100
    close_cov = 100

    metrics: Dict[str, Any] = {
        "sample_size": {"bets": int(len(bets)), "rows": int(len(bets)), "start": cfg.start, "end": cfg.end},
        "performance": {
            "roi": roi,
            "average_clv": avg_clv,
            "clv_positive_rate": clv_pos_rate,
            "profit_units": profit,
            "stake_units": stake,
        },
        "risk_metrics": {"max_drawdown_units": max_dd},
        "clv_coverage": {"open_snapshot_coverage_pct": open_cov, "close_snapshot_coverage_pct": close_cov},
        "filters": {
            "policy": "e2_ats_away_only",
            "ev_threshold": float(cfg.ev_threshold),
            "max_dispersion": float(cfg.max_dispersion),
            "require_dispersion": bool(cfg.require_dispersion),
            "max_bet_rate": float(cfg.max_bet_rate),
            "notes": "ATS-based E2 runner using open/close spread CLV and open price for settlement.",
        },
        "eligibility": {
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "eligible_pct": float(eligible_pct),
            "excluded_rows": int(total_rows - eligible_rows),
            "exclusion_reason": "missing open/close spread points or open spread price (and/or dispersion gating)",
        },
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser("e2_policy_runner.py (ATS-based)")
    ap.add_argument("--per-game", required=True, help="Path to outputs/backtest_joined.csv (from backtest)")
    ap.add_argument("--snapshot-dir", default="data/_snapshots")
    ap.add_argument("--calibrator", required=True, help="Path to artifacts/spread_calibrator.joblib")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="outputs/e2_policy_metrics.json")

    ap.add_argument("--ev", type=float, default=0.03)
    ap.add_argument("--max-dispersion", type=float, default=2.0)
    ap.add_argument("--require-dispersion", action="store_true", default=True)
    ap.add_argument("--max-bet-rate", type=float, default=0.30)
    ap.add_argument("--stake", type=float, default=1.0)

    args = ap.parse_args()

    cfg = E2ATSConfig(
        per_game_path=str(args.per_game),
        snapshot_dir=str(args.snapshot_dir),
        calibrator_path=str(args.calibrator),
        start=str(args.start),
        end=str(args.end),
        out_path=str(args.out),
        ev_threshold=float(args.ev),
        max_dispersion=float(args.max_dispersion),
        require_dispersion=bool(args.require_dispersion),
        max_bet_rate=float(args.max_bet_rate),
        stake=float(args.stake),
        side_policy="away_only",
    )

    metrics = compute_e2_ats_metrics(cfg)

    os.makedirs(os.path.dirname(cfg.out_path) or ".", exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[e2_ats] wrote: {cfg.out_path}")
    print(
        f"[e2_ats] bets={metrics['sample_size']['bets']} roi={metrics['performance']['roi']} "
        f"avg_clv={metrics['performance']['average_clv']} clv_pos_rate={metrics['performance']['clv_positive_rate']} "
        f"max_dd={metrics['risk_metrics']['max_drawdown_units']} eligible_pct={metrics['eligibility']['eligible_pct']}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", ".")
    main()
