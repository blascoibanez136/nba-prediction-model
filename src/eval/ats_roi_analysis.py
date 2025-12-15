"""
ATS (spread) ROI analysis.

Contracts / assumptions:
- Spread prices are implicit (fixed -110)
- Profit per 1u if win = +0.9090909
- Loss per 1u if lose = -1.0
- Push = 0.0

Signal:
- residual = fair_spread_model - home_spread_consensus  (or spread_error fallback)
- calibrator maps residual -> P(home_covers)
- P(away_covers) = 1 - P(home_covers)

Selection:
- Choose side with best EV, if EV >= threshold
- Optional dispersion gating via home_spread_dispersion

Outputs:
- outputs/ats_roi_metrics.json
- outputs/ats_roi_bets.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.model.spread_relative_calibration import load_spread_calibrator, apply_spread_calibrator

ATS_ROI_VERSION = "ats_roi_v1_ev_cal_fixed_110_dispersion_gated_2025-12-15"

PPU_ATS_MINUS_110 = 100.0 / 110.0  # 0.9090909


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _find_score_cols(df: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("home_score", "away_score"),
        ("home_pts", "away_pts"),
        ("home_points", "away_points"),
        ("pts_home", "pts_away"),
    ]
    for h, a in candidates:
        if h in df.columns and a in df.columns:
            return h, a
    raise RuntimeError("[ats_roi] Missing score columns (home_score/away_score or equivalent).")


def _get_residual(df: pd.DataFrame) -> pd.Series:
    if "fair_spread_model" in df.columns and "home_spread_consensus" in df.columns:
        return pd.to_numeric(df["fair_spread_model"], errors="coerce") - pd.to_numeric(df["home_spread_consensus"], errors="coerce")
    if "spread_error" in df.columns:
        return pd.to_numeric(df["spread_error"], errors="coerce")
    raise RuntimeError("[ats_roi] Need fair_spread_model+home_spread_consensus OR spread_error.")


def expected_value_ats(p_win: Optional[float]) -> Optional[float]:
    if p_win is None:
        return None
    p = _to_float(p_win)
    if p is None or not (0.0 < p < 1.0):
        return None
    return p * PPU_ATS_MINUS_110 - (1.0 - p) * 1.0


@dataclass(frozen=True)
class ATSConfig:
    per_game_path: str
    calibrator_path: str
    ev_threshold: float = 0.01

    max_dispersion: float = 2.0
    require_dispersion: bool = True

    max_bet_rate: float = 0.35
    max_profit_abs: float = 10.0

    out_dir: str = "outputs"


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}
    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0).sum())
    roi = (profit / stake) if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean())
    return {"bets": int(len(bets)), "stake": stake, "profit": profit, "roi": roi, "win_rate": win_rate}


def main() -> None:
    ap = argparse.ArgumentParser("ats_roi_analysis.py")
    ap.add_argument("--per_game", required=True, help="Path to outputs/backtest_per_game.csv")
    ap.add_argument("--calibrator", required=True, help="Path to artifacts/spread_calibrator.joblib")
    ap.add_argument("--ev", type=float, default=0.01, help="EV threshold (units) for ATS selection")

    ap.add_argument("--max-dispersion", type=float, default=2.0)
    ap.add_argument("--no-require-dispersion", action="store_true", help="If set, dispersion gate becomes optional")

    ap.add_argument("--max-bet-rate", type=float, default=0.35)
    ap.add_argument("--max-profit-abs", type=float, default=10.0)

    args = ap.parse_args()

    cfg = ATSConfig(
        per_game_path=args.per_game,
        calibrator_path=args.calibrator,
        ev_threshold=float(args.ev),
        max_dispersion=float(args.max_dispersion),
        require_dispersion=(not bool(args.no_require_dispersion)),
        max_bet_rate=float(args.max_bet_rate),
        max_profit_abs=float(args.max_profit_abs),
        out_dir="outputs",
    )

    print(f"[ats] version={ATS_ROI_VERSION}")
    print(f"[ats] __file__={__file__}")
    print(f"[ats] cwd={os.getcwd()}")
    print(f"[ats] per_game={cfg.per_game_path}")
    print(f"[ats] calibrator={cfg.calibrator_path}")
    print(f"[ats] ev_threshold={cfg.ev_threshold:.4f}")
    print(f"[ats] dispersion<= {cfg.max_dispersion:.2f} require={cfg.require_dispersion}")
    print(f"[ats] guards: max_profit_abs={cfg.max_profit_abs} max_bet_rate={cfg.max_bet_rate}")
    print(f"[ats] pricing: fixed -110 ppu={PPU_ATS_MINUS_110:.6f}")

    if not os.path.exists(cfg.per_game_path):
        raise FileNotFoundError(f"[ats] per_game not found: {cfg.per_game_path}")
    if not os.path.exists(cfg.calibrator_path):
        raise FileNotFoundError(f"[ats] calibrator not found: {cfg.calibrator_path}")

    df = pd.read_csv(cfg.per_game_path)
    if df.empty:
        raise RuntimeError("[ats] per_game is empty")

    if "home_spread_consensus" not in df.columns:
        raise RuntimeError("[ats] Missing required column: home_spread_consensus")

    hcol, acol = _find_score_cols(df)
    df[hcol] = pd.to_numeric(df[hcol], errors="coerce")
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    df["home_spread_consensus"] = pd.to_numeric(df["home_spread_consensus"], errors="coerce")

    # residual
    df["spread_residual"] = _get_residual(df)

    # dispersion
    if "home_spread_dispersion" in df.columns:
        df["home_spread_dispersion"] = pd.to_numeric(df["home_spread_dispersion"], errors="coerce")
        disp_gate = df["home_spread_dispersion"].le(cfg.max_dispersion)
    else:
        df["home_spread_dispersion"] = pd.NA
        disp_gate = pd.Series([True] * len(df))

    if cfg.require_dispersion and "home_spread_dispersion" in df.columns:
        eligible = disp_gate & df["home_spread_dispersion"].notna()
    elif cfg.require_dispersion and "home_spread_dispersion" not in df.columns:
        raise RuntimeError("[ats] require_dispersion=True but home_spread_dispersion column is missing")
    else:
        eligible = disp_gate | df["home_spread_dispersion"].isna()

    # require core fields
    eligible = eligible & df["spread_residual"].notna() & df["home_spread_consensus"].notna() & df[hcol].notna() & df[acol].notna()

    df["eligible"] = eligible

    cal = load_spread_calibrator(cfg.calibrator_path)
    df["p_home_cover"] = df.apply(
        lambda r: apply_spread_calibrator(
            residual=_to_float(r["spread_residual"]),
            home_spread_consensus=_to_float(r["home_spread_consensus"]),
            calibrator=cal,
        ),
        axis=1,
    )
    df["p_away_cover"] = 1.0 - pd.to_numeric(df["p_home_cover"], errors="coerce")

    df["home_ev"] = df["p_home_cover"].apply(expected_value_ats)
    df["away_ev"] = df["p_away_cover"].apply(expected_value_ats)

    def choose_side(r) -> Tuple[bool, Optional[str], Optional[float]]:
        if not bool(r["eligible"]):
            return False, None, None
        hev = r.get("home_ev", None)
        aev = r.get("away_ev", None)
        if pd.isna(hev) and pd.isna(aev):
            return False, None, None

        # best-of; tie -> no bet (strict)
        if pd.notna(hev) and pd.notna(aev) and float(hev) == float(aev):
            return False, None, None

        if pd.notna(hev) and float(hev) >= cfg.ev_threshold and (pd.isna(aev) or float(hev) > float(aev)):
            return True, "home", float(hev)
        if pd.notna(aev) and float(aev) >= cfg.ev_threshold and (pd.isna(hev) or float(aev) > float(hev)):
            return True, "away", float(aev)

        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0].astype(bool)
    df["bet_side"] = chosen[1]
    df["ev_used"] = chosen[2]

    bets = df[df["bet"]].copy()
    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    bet_rate = float(len(bets) / max(total_games, 1))

    # Fail-loud bet-rate regression detector
    if bet_rate > cfg.max_bet_rate:
        raise RuntimeError(
            f"[ats] Bet-rate too high: bets={len(bets)} total_games={total_games} bet_rate={bet_rate:.3f} cap={cfg.max_bet_rate}"
        )

    if bets.empty:
        print("[ats] No bets selected.")
        out_metrics = {
            "version": ATS_ROI_VERSION,
            "bets": 0,
            "bet_rate": bet_rate,
            "note": "No bets met EV threshold / gating.",
        }
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "ats_roi_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(out_metrics, f, indent=2)
        bets.to_csv(os.path.join(cfg.out_dir, "ats_roi_bets.csv"), index=False)
        print(f"[ats] wrote: {os.path.join(cfg.out_dir, 'ats_roi_metrics.json')}")
        print(f"[ats] wrote: {os.path.join(cfg.out_dir, 'ats_roi_bets.csv')}")
        return

    bets["stake"] = 1.0

    # settle ATS result
    def ats_result(r) -> str:
        hs = float(r[hcol])
        aw = float(r[acol])
        line = float(r["home_spread_consensus"])
        adj_home = hs + line
        if adj_home == aw:
            return "push"
        home_covers = adj_home > aw
        side = str(r["bet_side"]).lower()
        if side == "home":
            return "win" if home_covers else "loss"
        if side == "away":
            return "win" if (not home_covers) else "loss"
        return "unknown"

    bets["result"] = bets.apply(ats_result, axis=1)

    def profit_units(r) -> float:
        res = str(r["result"]).lower()
        if res == "push":
            return 0.0
        if res != "win":
            return -1.0
        return float(PPU_ATS_MINUS_110)

    bets["profit"] = bets.apply(profit_units, axis=1)
    bets["market"] = "spread"

    # Profit sanity
    max_abs = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs > cfg.max_profit_abs:
        raise RuntimeError(f"[ats] Profit sanity failure: max |profit|={max_abs}u limit={cfg.max_profit_abs}")

    overall = summarize(bets)
    home_bets = bets[bets["bet_side"].astype(str).str.lower() == "home"].copy()
    away_bets = bets[bets["bet_side"].astype(str).str.lower() == "away"].copy()
    home_sum = summarize(home_bets)
    away_sum = summarize(away_bets)

    print(f"[ats] overall: {overall}")
    print(f"[ats] home_only: {home_sum}")
    print(f"[ats] away_only: {away_sum}")
    print(f"[ats] bet_rate: {bet_rate:.3f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.out_dir, "ats_roi_metrics.json")
    bets_path = os.path.join(cfg.out_dir, "ats_roi_bets.csv")

    metrics: Dict[str, Any] = {
        "version": ATS_ROI_VERSION,
        "overall": overall,
        "home_only": home_sum,
        "away_only": away_sum,
        "bet_rate": bet_rate,
        "ev_threshold": cfg.ev_threshold,
        "dispersion": {
            "max_dispersion": cfg.max_dispersion,
            "require_dispersion": cfg.require_dispersion,
            "col": "home_spread_dispersion" if "home_spread_dispersion" in df.columns else None,
        },
        "pricing": {"assumed": "-110", "ppu": PPU_ATS_MINUS_110},
        "calibrator": cfg.calibrator_path,
        "schema_contract": {
            "required_cols": ["home_spread_consensus", "fair_spread_model|spread_error", hcol, acol],
            "merge_key_present": ("merge_key" in df.columns),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    bets.to_csv(bets_path, index=False)

    print(f"[ats] wrote: {metrics_path}")
    print(f"[ats] wrote: {bets_path}")


if __name__ == "__main__":
    main()
