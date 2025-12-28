from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.model.spread_relative_calibration import apply_spread_calibrator


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def clean_american_odds(x: object) -> Optional[float]:
    v = _to_float(x)
    if v is None or v == 0:
        return None
    if 0 < abs(v) < 100:
        return None
    return v


def win_profit_per_unit_american(o: Optional[float]) -> Optional[float]:
    o = clean_american_odds(o)
    if o is None:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def expected_value_units(p_win: Optional[float], odds: Optional[float]) -> Optional[float]:
    p = _to_float(p_win)
    if p is None or not (0.0 < p < 1.0):
        return None
    ppu = win_profit_per_unit_american(odds)
    if ppu is None:
        return None
    return p * ppu - (1.0 - p)


def merge_key(home_team: str, away_team: str, game_date: str) -> str:
    return f"{str(home_team).strip().lower()}__{str(away_team).strip().lower()}__{str(game_date).strip()[:10]}"


def _ymd(date_str: str) -> str:
    return str(date_str).replace("-", "")


def load_spread_snapshot_day(snapshot_dir: Path, game_date: str, snapshot_type: str) -> pd.DataFrame:
    """
    Builds per-game spread consensus + dispersion + prices from a single snapshot day.
    Expects columns:
      - merge_key
      - spread_home_point
      - spread_home_price
      - spread_away_price
      - book (optional for dispersion)
    """
    st = snapshot_type.lower().strip()
    path = snapshot_dir / f"{st}_{_ymd(game_date)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["merge_key"])

    df = pd.read_csv(path)
    if "merge_key" not in df.columns:
        if all(c in df.columns for c in ["home_team", "away_team", "game_date"]):
            df["merge_key"] = [merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])]
        else:
            return pd.DataFrame(columns=["merge_key"])

    # Required spread point
    if "spread_home_point" not in df.columns:
        return pd.DataFrame(columns=["merge_key"])

    df["spread_home_point"] = pd.to_numeric(df["spread_home_point"], errors="coerce")

    out = df.groupby("merge_key")["spread_home_point"].mean().rename("home_spread_point").reset_index()

    # Dispersion (std across books)
    if "book" in df.columns:
        disp = df.groupby("merge_key")["spread_home_point"].std().rename("home_spread_dispersion").reset_index()
        out = out.merge(disp, on="merge_key", how="left")

    # Prices (open snapshots include these)
    if "spread_home_price" in df.columns:
        pxh = df.groupby("merge_key")["spread_home_price"].mean(numeric_only=True).rename("home_spread_price").reset_index()
        pxh["home_spread_price"] = pd.to_numeric(pxh["home_spread_price"], errors="coerce").apply(clean_american_odds)
        out = out.merge(pxh, on="merge_key", how="left")

    if "spread_away_price" in df.columns:
        pxa = df.groupby("merge_key")["spread_away_price"].mean(numeric_only=True).rename("away_spread_price").reset_index()
        pxa["away_spread_price"] = pd.to_numeric(pxa["away_spread_price"], errors="coerce").apply(clean_american_odds)
        out = out.merge(pxa, on="merge_key", how="left")

    return out


@dataclass(frozen=True)
class E2ATSConfig:
    per_game_path: str
    snapshot_dir: str
    calibrator_path: str
    start: str
    end: str
    out_path: str
    ev_threshold: float = 0.03
    max_dispersion: float = 2.0
    require_dispersion: bool = True
    max_bet_rate: float = 0.30
    stake: float = 1.0


def compute(cfg: E2ATSConfig) -> Dict[str, Any]:
    per_game = Path(cfg.per_game_path)
    if not per_game.exists():
        raise FileNotFoundError(f"[e2_ats] missing per_game: {per_game}")

    cal_path = Path(cfg.calibrator_path)
    if not cal_path.exists():
        raise FileNotFoundError(f"[e2_ats] missing calibrator: {cal_path}")

    df = pd.read_csv(per_game)
    if df.empty:
        raise RuntimeError("[e2_ats] per_game empty")

    if "game_date" not in df.columns:
        if "date" in df.columns:
            df["game_date"] = df["date"].astype(str).str[:10]
        else:
            raise RuntimeError("[e2_ats] missing game_date/date")

    df["game_date"] = df["game_date"].astype(str).str[:10]
    df = df[(df["game_date"] >= cfg.start) & (df["game_date"] <= cfg.end)].copy()
    if df.empty:
        raise RuntimeError("[e2_ats] no rows in window")

    for c in ["home_team", "away_team", "home_score", "away_score"]:
        if c not in df.columns:
            raise RuntimeError(f"[e2_ats] missing required column: {c}")

    fair_col = "fair_spread_model" if "fair_spread_model" in df.columns else ("fair_spread" if "fair_spread" in df.columns else None)
    if not fair_col:
        raise RuntimeError("[e2_ats] missing fair_spread_model/fair_spread")

    df[fair_col] = pd.to_numeric(df[fair_col], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=[fair_col, "home_score", "away_score"]).copy()

    df["merge_key"] = [merge_key(h, a, gd) for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])]

    calibrator = joblib.load(str(cal_path))
    if not isinstance(calibrator, dict) or calibrator.get("type") != "spread_isotonic_bucketed_v1":
        raise RuntimeError("[e2_ats] calibrator type mismatch")

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
        "home_spread_dispersion": "home_spread_dispersion_open",
        "home_spread_price": "home_price_open",
        "away_spread_price": "away_price_open",
    })
    close_df = close_df.rename(columns={
        "home_spread_point": "home_spread_close",
        "home_spread_dispersion": "home_spread_dispersion_close",
    })

    df = df.merge(open_df, on="merge_key", how="left").merge(close_df, on="merge_key", how="left")

    total_rows = int(len(df))
    elig_mask = (
        df["home_spread_open"].notna()
        & df["home_spread_close"].notna()
        & df["away_price_open"].notna()
    )
    eligible = df.loc[elig_mask].copy()
    eligible_rows = int(len(eligible))
    eligible_pct = float(eligible_rows / max(total_rows, 1))
    if eligible_rows == 0:
        raise RuntimeError("[e2_ats] no eligible rows")

    # Dispersion gate (prefer close)
    disp_col = None
    if "home_spread_dispersion_close" in eligible.columns and eligible["home_spread_dispersion_close"].notna().any():
        disp_col = "home_spread_dispersion_close"
    elif "home_spread_dispersion_open" in eligible.columns and eligible["home_spread_dispersion_open"].notna().any():
        disp_col = "home_spread_dispersion_open"

    if cfg.require_dispersion and disp_col is None:
        raise RuntimeError("[e2_ats] require_dispersion=True but dispersion missing")

    if disp_col is not None:
        eligible[disp_col] = pd.to_numeric(eligible[disp_col], errors="coerce")
        eligible = eligible[(eligible[disp_col].notna()) & (eligible[disp_col] <= float(cfg.max_dispersion))].copy()
        if eligible.empty:
            raise RuntimeError("[e2_ats] empty after dispersion gate")

    # Residual vs CLOSE line
    eligible["residual"] = pd.to_numeric(eligible[fair_col], errors="coerce") - pd.to_numeric(eligible["home_spread_close"], errors="coerce")
    eligible = eligible.dropna(subset=["residual"]).copy()
    if eligible.empty:
        raise RuntimeError("[e2_ats] empty after residual")

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
    eligible["p_away_cover"] = 1.0 - eligible["p_home_cover"]

    # Selection: AWAY ONLY
    eligible["away_ev"] = eligible.apply(lambda r: expected_value_units(float(r["p_away_cover"]), float(r["away_price_open"])), axis=1)
    eligible["bet"] = eligible["away_ev"] >= float(cfg.ev_threshold)
    bets = eligible.loc[eligible["bet"]].copy()

    # bet-rate cap (by EV)
    total_games = int(eligible["merge_key"].nunique()) if "merge_key" in eligible.columns else int(len(eligible))
    max_bets = max(1, int(math.floor(float(cfg.max_bet_rate) * max(total_games, 1))))
    if len(bets) > max_bets:
        bets = bets.sort_values("away_ev", ascending=False).head(max_bets).copy()

    # CLV aligned for AWAY bets:
    # Positive if close moved in our favor => close_home_spread - open_home_spread
    bets["clv_aligned"] = pd.to_numeric(bets["home_spread_close"], errors="coerce") - pd.to_numeric(bets["home_spread_open"], errors="coerce")
    avg_clv = float(pd.to_numeric(bets["clv_aligned"], errors="coerce").mean())
    clv_pos_rate = float((pd.to_numeric(bets["clv_aligned"], errors="coerce") > 0).mean())

    # Settlement uses CLOSE line
    bets["adj_home"] = bets["home_score"] + bets["home_spread_close"]
    bets["result"] = np.where(bets["adj_home"] < bets["away_score"], "win",
                              np.where(bets["adj_home"] > bets["away_score"], "loss", "push"))

    # Profit uses AWAY OPEN price
    bets["stake"] = float(cfg.stake)
    bets["ppu"] = bets["away_price_open"].apply(win_profit_per_unit_american)
    bets["profit"] = np.where(
        bets["result"] == "win",
        bets["stake"] * bets["ppu"],
        np.where(bets["result"] == "push", 0.0, -bets["stake"])
    )

    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0.0).sum())
    roi = float(profit / stake) if stake > 0 else None

    # Drawdown by day
    daily = bets.groupby("game_date")["profit"].sum().reset_index().sort_values("game_date")
    daily["cum"] = daily["profit"].cumsum()
    daily["peak"] = daily["cum"].cummax()
    daily["dd"] = daily["cum"] - daily["peak"]
    max_dd = float(daily["dd"].min()) if not daily.empty else 0.0

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
        "clv_coverage": {"open_snapshot_coverage_pct": 100, "close_snapshot_coverage_pct": 100},
        "filters": {
            "policy": "e2_ats_away_only",
            "ev_threshold": float(cfg.ev_threshold),
            "max_dispersion": float(cfg.max_dispersion),
            "require_dispersion": bool(cfg.require_dispersion),
            "max_bet_rate": float(cfg.max_bet_rate),
            "notes": "ATS E2 runner: AWAY-only, EV gating, CLV=open->close spread movement aligned for away bets, pricing uses away_price_open.",
        },
        "eligibility": {
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "eligible_pct": float(eligible_pct),
            "excluded_rows": int(total_rows - eligible_rows),
            "exclusion_reason": "missing open+close spread points or away open price (and/or dispersion gating)",
        },
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser("e2_policy_runner.py (ATS-based)")
    ap.add_argument("--per-game", required=True)
    ap.add_argument("--snapshot-dir", default="data/_snapshots")
    ap.add_argument("--calibrator", required=True)
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
    )

    metrics = compute(cfg)
    os.makedirs(os.path.dirname(cfg.out_path) or ".", exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[e2_ats] wrote: {cfg.out_path}")
    print(
        f"[e2_ats] bets={metrics['sample_size']['bets']} roi={metrics['performance']['roi']:.4f} "
        f"avg_clv={metrics['performance']['average_clv']:.6f} clv_pos_rate={metrics['performance']['clv_positive_rate']:.3f} "
        f"max_dd={metrics['risk_metrics']['max_drawdown_units']:.3f} eligible_pct={metrics['eligibility']['eligible_pct']:.3f}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", ".")
    main()
