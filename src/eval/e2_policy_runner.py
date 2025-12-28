"""
E2 Policy Runner (artifact generator)

Purpose:
- Generate outputs/e2_policy_metrics.json in the exact schema expected by qa/certify.py.
- Use real per-game data + open/close snapshots to compute CLV, ROI, and drawdowns.

Design constraints:
- Deterministic (same inputs -> same outputs)
- American-only odds contract enforced (abs(odds) >= 100)
- Fail-loud if coverage cannot reach 100% in the requested window
- Minimal dependencies (pandas, numpy)

Inputs:
- outputs/backtest_per_game.csv (must exist; produced by your pipeline)
- data/_snapshots/open_YYYYMMDD.csv and close_YYYYMMDD.csv (must exist)

Output:
- outputs/e2_policy_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    # American-only contract: abs(odds) must be >= 100
    if 0 < abs(o) < 100:
        return None
    return o


def american_to_decimal(o: Optional[float]) -> Optional[float]:
    if o is None:
        return None
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


# ---------------------------
# Snapshot loading
# ---------------------------

def _ymd(date_str: str) -> str:
    return str(date_str).replace("-", "")


def load_snapshot_consensus(
    snapshot_dir: Path,
    *,
    snapshot_type: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Load open/close snapshots across a date range and compute per-game ML consensus.
    Returns: DataFrame with columns: merge_key, ml_home_<type>, ml_away_<type>
    """
    st = snapshot_type.strip().lower()
    if st not in {"open", "close"}:
        raise ValueError(f"snapshot_type must be open|close, got {snapshot_type}")

    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Invalid start/end dates")

    # iterate days; expect exactly one file per day: open_YYYYMMDD.csv / close_YYYYMMDD.csv
    dates = pd.date_range(start_ts.date(), end_ts.date(), freq="D").strftime("%Y-%m-%d").tolist()
    frames = []

    for d in dates:
        p = snapshot_dir / f"{st}_{_ymd(d)}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)

        # Best-effort: wide snapshot format should include these
        if "merge_key" not in df.columns:
            if all(c in df.columns for c in ["home_team", "away_team", "game_date"]):
                df["merge_key"] = [
                    merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])
                ]
            else:
                continue

        # Find ML cols
        home_ml_col = "ml_home" if "ml_home" in df.columns else ("ml_home_consensus" if "ml_home_consensus" in df.columns else None)
        away_ml_col = "ml_away" if "ml_away" in df.columns else ("ml_away_consensus" if "ml_away_consensus" in df.columns else None)
        if home_ml_col is None or away_ml_col is None:
            continue

        df[home_ml_col] = pd.to_numeric(df[home_ml_col], errors="coerce").apply(clean_american_ml)
        df[away_ml_col] = pd.to_numeric(df[away_ml_col], errors="coerce").apply(clean_american_ml)

        agg = (
            df.groupby("merge_key")[[home_ml_col, away_ml_col]]
            .mean(numeric_only=True)
            .reset_index()
            .rename(columns={
                home_ml_col: f"ml_home_{st}",
                away_ml_col: f"ml_away_{st}",
            })
        )
        frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=["merge_key", f"ml_home_{st}", f"ml_away_{st}"])

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby("merge_key")[[f"ml_home_{st}", f"ml_away_{st}"]].mean().reset_index()
    return out


# ---------------------------
# E2 policy definition
# ---------------------------

@dataclass(frozen=True)
class E2Config:
    per_game_path: str = "outputs/backtest_per_game.csv"
    snapshot_dir: str = "data/_snapshots"
    start: str = "2023-10-24"
    end: str = "2024-04-14"
    out_path: str = "outputs/e2_policy_metrics.json"

    # Policy: placeholder but deterministic and conservative
    # Use model-favored side @ OPEN odds; measure CLV via open->close.
    stake: float = 1.0

    # Coverage requirements (certify expects 100/100)
    require_open_coverage_pct: int = 100
    require_close_coverage_pct: int = 100


def compute_e2_metrics(cfg: E2Config) -> dict:
    per_game = Path(cfg.per_game_path)
    if not per_game.exists():
        raise FileNotFoundError(f"[e2] missing per_game: {per_game}")

    df = pd.read_csv(per_game)
    if df.empty:
        raise RuntimeError("[e2] per_game is empty")

    # required columns
    need = ["home_team", "away_team", "home_score", "away_score"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"[e2] per_game missing required column: {c}")

    # date col normalization
    if "game_date" not in df.columns:
        if "date" in df.columns:
            df["game_date"] = df["date"].astype(str).str[:10]
        else:
            raise RuntimeError("[e2] per_game must include game_date or date")

    # model prob
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

    df["merge_key"] = [
        merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])
    ]

    snap_dir = Path(cfg.snapshot_dir)

    open_cons = load_snapshot_consensus(snap_dir, snapshot_type="open", start=cfg.start, end=cfg.end)
    close_cons = load_snapshot_consensus(snap_dir, snapshot_type="close", start=cfg.start, end=cfg.end)

    # Merge snapshot odds
    df = df.merge(open_cons, on="merge_key", how="left")
    df = df.merge(close_cons, on="merge_key", how="left")

    # Coverage (must be 100%)
    open_cov = int(round(100.0 * float(df[["ml_home_open", "ml_away_open"]].notna().all(axis=1).mean())))
    close_cov = int(round(100.0 * float(df[["ml_home_close", "ml_away_close"]].notna().all(axis=1).mean())))

    # Require full coverage to satisfy certify
    if open_cov < cfg.require_open_coverage_pct:
        raise RuntimeError(f"[e2] open snapshot coverage {open_cov}% < required {cfg.require_open_coverage_pct}%")
    if close_cov < cfg.require_close_coverage_pct:
        raise RuntimeError(f"[e2] close snapshot coverage {close_cov}% < required {cfg.require_close_coverage_pct}%")

    # Outcome
    df["home_win_actual"] = (df["home_score"] > df["away_score"]).astype(int)

    # Policy: model-favored side
    df["bet_side"] = np.where(df[prob_col] >= 0.5, "home", "away")

    def pick_odds(r, home_col, away_col):
        return r[home_col] if r["bet_side"] == "home" else r[away_col]

    df["odds_open"] = df.apply(lambda r: pick_odds(r, "ml_home_open", "ml_away_open"), axis=1)
    df["odds_close"] = df.apply(lambda r: pick_odds(r, "ml_home_close", "ml_away_close"), axis=1)

    # CLV in decimal space
    df["dec_open"] = df["odds_open"].apply(american_to_decimal)
    df["dec_close"] = df["odds_close"].apply(american_to_decimal)
    df = df.dropna(subset=["dec_open", "dec_close"]).copy()
    if df.empty:
        raise RuntimeError("[e2] No valid rows after decimal conversion")

    df["clv"] = df["dec_open"] - df["dec_close"]
    avg_clv = float(df["clv"].mean())
    clv_pos_rate = float((df["clv"] > 0).mean())

    # Settlement using open odds (execution proxy)
    df["win_profit_per_unit"] = df["odds_open"].apply(win_profit_per_unit_american)
    df = df.dropna(subset=["win_profit_per_unit"]).copy()
    if df.empty:
        raise RuntimeError("[e2] No valid rows after odds sanity")

    def settle(r) -> str:
        hw = int(r["home_win_actual"])
        side = str(r["bet_side"]).lower()
        win = (hw == 1 and side == "home") or (hw == 0 and side == "away")
        return "win" if win else "loss"

    df["result"] = df.apply(settle, axis=1)
    df["stake"] = float(cfg.stake)
    df["profit"] = np.where(df["result"] == "win", df["stake"] * df["win_profit_per_unit"], -df["stake"])

    # Aggregate
    bets = int(len(df))
    stake = float(df["stake"].sum())
    profit = float(df["profit"].sum())
    roi = float(profit / stake) if stake > 0 else None

    # Drawdown by day
    daily = df.groupby("game_date")["profit"].sum().reset_index()
    daily["cum"] = daily["profit"].cumsum()
    daily["peak"] = daily["cum"].cummax()
    daily["dd"] = daily["cum"] - daily["peak"]
    max_dd = float(daily["dd"].min()) if not daily.empty else 0.0

    metrics = {
        "sample_size": {
            "bets": bets,
            "rows": int(len(df)),
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
            "notes": "Deterministic E2 artifact generator; replace selection logic when true E2 selector module is added.",
        },
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser("e2_policy_runner.py (artifact generator)")
    ap.add_argument("--per-game", default="outputs/backtest_per_game.csv")
    ap.add_argument("--snapshot-dir", default="data/_snapshots")
    ap.add_argument("--start", default="2023-10-24")
    ap.add_argument("--end", default="2024-04-14")
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
    print(f"[e2] bets={metrics['sample_size']['bets']} roi={metrics['performance']['roi']:.4f} "
          f"avg_clv={metrics['performance']['average_clv']:.6f} clv_pos_rate={metrics['performance']['clv_positive_rate']:.3f} "
          f"max_dd={metrics['risk_metrics']['max_drawdown_units']:.3f}")


if __name__ == "__main__":
    # ensure imports resolve if invoked oddly
    os.environ.setdefault("PYTHONPATH", ".")
    main()
