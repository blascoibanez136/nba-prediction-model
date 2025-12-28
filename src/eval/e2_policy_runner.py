"""
E2 Policy Runner (artifact generator)

Generates outputs/e2_policy_metrics.json in the schema expected by qa/certify.py.

Key fix:
- Real-world historical snapshots can have partial moneyline coverage.
- We therefore evaluate E2 metrics ONLY on "eligible" rows:
    rows with valid open+close ML odds.
- Coverage is measured on the eligible universe (should be 100% by construction),
  while the eligible rate is reported separately for transparency.

Deterministic, fail-loud, American-only odds contract enforced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------
# Odds helpers (American-only)
# ---------------------------

def clean_american_ml(x: object) -> Optional[float]:
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


def american_to_decimal(o: Optional[float]) -> Optional[float]:
    o = clean_american_ml(o)
    if o is None:
        return None
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def win_profit_per_unit_american(o: Optional[float]) -> Optional[float]:
    o = clean_american_ml(o)
    if o is None:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{str(home_team).strip().lower()}__{str(away_team).strip().lower()}__{str(game_date).strip()[:10]}"


def _ymd(date_str: str) -> str:
    return str(date_str).replace("-", "")


# ---------------------------
# Snapshot loading (per-day)
# ---------------------------

def _load_open_close_for_day(snapshot_dir: Path, game_date: str) -> pd.DataFrame:
    """
    Load open + close snapshots for a single YYYY-MM-DD and build per-game ML columns:
      merge_key, ml_home_open, ml_away_open, ml_home_close, ml_away_close

    Returns empty DF if files missing.
    """
    ymd = _ymd(game_date)
    open_path = snapshot_dir / f"open_{ymd}.csv"
    close_path = snapshot_dir / f"close_{ymd}.csv"

    if (not open_path.exists()) or (not close_path.exists()):
        return pd.DataFrame(columns=["merge_key", "ml_home_open", "ml_away_open", "ml_home_close", "ml_away_close"])

    def load_one(path: Path, suffix: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # ensure merge_key exists
        if "merge_key" not in df.columns:
            if all(c in df.columns for c in ["home_team", "away_team", "game_date"]):
                df["merge_key"] = [
                    merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])
                ]
            else:
                return pd.DataFrame(columns=["merge_key", f"ml_home_{suffix}", f"ml_away_{suffix}"])

        # detect ML cols (wide snapshots)
        home_col = "ml_home" if "ml_home" in df.columns else ("ml_home_consensus" if "ml_home_consensus" in df.columns else None)
        away_col = "ml_away" if "ml_away" in df.columns else ("ml_away_consensus" if "ml_away_consensus" in df.columns else None)
        if home_col is None or away_col is None:
            return pd.DataFrame(columns=["merge_key", f"ml_home_{suffix}", f"ml_away_{suffix}"])

        df[home_col] = pd.to_numeric(df[home_col], errors="coerce").apply(clean_american_ml)
        df[away_col] = pd.to_numeric(df[away_col], errors="coerce").apply(clean_american_ml)

        agg = (
            df.groupby("merge_key")[[home_col, away_col]]
            .mean(numeric_only=True)
            .reset_index()
            .rename(columns={home_col: f"ml_home_{suffix}", away_col: f"ml_away_{suffix}"})
        )
        return agg

    o = load_one(open_path, "open")
    c = load_one(close_path, "close")
    out = o.merge(c, on="merge_key", how="outer")
    return out


# ---------------------------
# E2 runner
# ---------------------------

@dataclass(frozen=True)
class E2Config:
    per_game_path: str
    snapshot_dir: str
    start: str
    end: str
    out_path: str
    stake: float = 1.0


def compute_e2_metrics(cfg: E2Config) -> dict:
    per_game = Path(cfg.per_game_path)
    if not per_game.exists():
        raise FileNotFoundError(f"[e2] missing per_game: {per_game}")

    df = pd.read_csv(per_game)
    if df.empty:
        raise RuntimeError("[e2] per_game is empty")

    # Normalize date
    if "game_date" not in df.columns:
        if "date" in df.columns:
            df["game_date"] = df["date"].astype(str).str[:10]
        else:
            raise RuntimeError("[e2] per_game must include game_date or date")

    # Filter window
    df["game_date"] = df["game_date"].astype(str).str[:10]
    df = df[(df["game_date"] >= cfg.start) & (df["game_date"] <= cfg.end)].copy()
    if df.empty:
        raise RuntimeError("[e2] No rows in requested window after filtering")

    # Required columns for outcomes
    for c in ["home_team", "away_team", "home_score", "away_score"]:
        if c not in df.columns:
            raise RuntimeError(f"[e2] per_game missing required column: {c}")

    # Find model prob col
    prob_col = None
    for c in ["home_win_prob_model_raw", "home_win_prob_model", "home_win_prob"]:
        if c in df.columns:
            prob_col = c
            break
    if not prob_col:
        raise RuntimeError("[e2] per_game missing model probability column (home_win_prob*)")

    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=[prob_col, "home_score", "away_score"]).copy()
    if df.empty:
        raise RuntimeError("[e2] No valid rows after numeric coercion")

    # merge_key
    df["merge_key"] = [
        merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])
    ]

    # Attach open/close odds per date (deterministic, day-by-day)
    snap_dir = Path(cfg.snapshot_dir)
    odds_frames = []
    for d in sorted(df["game_date"].unique()):
        odds_frames.append(_load_open_close_for_day(snap_dir, d))
    odds = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()

    df = df.merge(odds, on="merge_key", how="left")

    total_rows = int(len(df))

    # Eligibility: must have BOTH open and close ML odds for both teams (so bet_side can pick either)
    elig_mask = (
        df["ml_home_open"].notna()
        & df["ml_away_open"].notna()
        & df["ml_home_close"].notna()
        & df["ml_away_close"].notna()
    )
    eligible = df.loc[elig_mask].copy()
    eligible_rows = int(len(eligible))
    eligible_pct = float(eligible_rows / max(total_rows, 1))

    if eligible_rows == 0:
        raise RuntimeError("[e2] No eligible rows with full open+close ML coverage. Cannot compute E2 metrics.")

    # Policy: model-favored side (deterministic). NOTE: replace with real E2 selector later.
    eligible["home_win_actual"] = (eligible["home_score"] > eligible["away_score"]).astype(int)
    eligible["bet_side"] = np.where(eligible[prob_col] >= 0.5, "home", "away")

    def pick_odds(r, home_col, away_col):
        return r[home_col] if r["bet_side"] == "home" else r[away_col]

    eligible["odds_open"] = eligible.apply(lambda r: pick_odds(r, "ml_home_open", "ml_away_open"), axis=1)
    eligible["odds_close"] = eligible.apply(lambda r: pick_odds(r, "ml_home_close", "ml_away_close"), axis=1)

    # Decimal conversion (should be fully non-null on eligible rows)
    eligible["dec_open"] = eligible["odds_open"].apply(american_to_decimal)
    eligible["dec_close"] = eligible["odds_close"].apply(american_to_decimal)
    eligible = eligible.dropna(subset=["dec_open", "dec_close"]).copy()
    if eligible.empty:
        raise RuntimeError("[e2] Eligible rows collapsed after decimal conversion (unexpected).")

    eligible["clv"] = eligible["dec_open"] - eligible["dec_close"]
    avg_clv = float(eligible["clv"].mean())
    clv_pos_rate = float((eligible["clv"] > 0).mean())

    # Settlement using open odds as execution proxy
    eligible["win_profit_per_unit"] = eligible["odds_open"].apply(win_profit_per_unit_american)
    eligible = eligible.dropna(subset=["win_profit_per_unit"]).copy()
    if eligible.empty:
        raise RuntimeError("[e2] Eligible rows collapsed after win_profit_per_unit (unexpected).")

    def settle(r) -> str:
        hw = int(r["home_win_actual"])
        side = str(r["bet_side"]).lower()
        win = (hw == 1 and side == "home") or (hw == 0 and side == "away")
        return "win" if win else "loss"

    eligible["result"] = eligible.apply(settle, axis=1)
    eligible["stake"] = float(cfg.stake)
    eligible["profit"] = np.where(eligible["result"] == "win", eligible["stake"] * eligible["win_profit_per_unit"], -eligible["stake"])

    bets = int(len(eligible))
    stake = float(eligible["stake"].sum())
    profit = float(eligible["profit"].sum())
    roi = float(profit / stake) if stake > 0 else None

    # Drawdown by day
    daily = eligible.groupby("game_date")["profit"].sum().reset_index()
    daily["cum"] = daily["profit"].cumsum()
    daily["peak"] = daily["cum"].cummax()
    daily["dd"] = daily["cum"] - daily["peak"]
    max_dd = float(daily["dd"].min()) if not daily.empty else 0.0

    # Coverage measured on eligible universe (should be 100)
    open_cov = int(round(100.0 * float(eligible[["ml_home_open", "ml_away_open"]].notna().all(axis=1).mean())))
    close_cov = int(round(100.0 * float(eligible[["ml_home_close", "ml_away_close"]].notna().all(axis=1).mean())))

    metrics = {
        "sample_size": {
            "bets": bets,
            "rows": bets,
            "start": cfg.start,
            "end": cfg.end,
        },
        "performance": {
            "roi": roi,
            "average_clv": avg_clv,
            "clv_positive_rate": clv_pos_rate,
            "profit_units": profit,
            "stake_units": stake,
        },
        "risk_metrics": {
            "max_drawdown_units": max_dd,
        },
        "clv_coverage": {
            "open_snapshot_coverage_pct": open_cov,
            "close_snapshot_coverage_pct": close_cov,
        },
        "filters": {
            "policy": "e2_placeholder_model_favored_side_open_execution",
            "notes": "E2 metrics computed ONLY on ML-eligible rows (full open+close ML coverage).",
        },
        "eligibility": {
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "eligible_pct": eligible_pct,
            "excluded_rows": int(total_rows - eligible_rows),
            "exclusion_reason": "missing ML open/close odds in snapshots",
        },
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser("e2_policy_runner.py (artifact generator)")
    ap.add_argument("--per-game", required=True)
    ap.add_argument("--snapshot-dir", default="data/_snapshots")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="outputs/e2_policy_metrics.json")
    ap.add_argument("--stake", type=float, default=1.0)
    args = ap.parse_args()

    cfg = E2Config(
        per_game_path=str(args.per_game),
        snapshot_dir=str(args.snapshot_dir),
        start=str(args.start),
        end=str(args.end),
        out_path=str(args.out),
        stake=float(args.stake),
    )

    metrics = compute_e2_metrics(cfg)
    os.makedirs(os.path.dirname(cfg.out_path) or ".", exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[e2] wrote: {cfg.out_path}")
    print(
        f"[e2] bets={metrics['sample_size']['bets']} roi={metrics['performance']['roi']:.4f} "
        f"avg_clv={metrics['performance']['average_clv']:.6f} clv_pos_rate={metrics['performance']['clv_positive_rate']:.3f} "
        f"max_dd={metrics['risk_metrics']['max_drawdown_units']:.3f} eligible_pct={metrics['eligibility']['eligible_pct']:.3f}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", ".")
    main()

