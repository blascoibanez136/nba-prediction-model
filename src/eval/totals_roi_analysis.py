"""
Totals (Over/Under) ROI analysis with fixed -110 pricing.

PACKET 4A-3:
- Residual magnitude gating
- Optional dispersion gating
- EV × residual interaction (funnel)
- Hard bet-rate + profit guards
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional

import pandas as pd
import joblib

from src.model.total_ev import expected_value_total

TOTALS_ROI_VERSION = "totals_roi_v2_ev_residual_funnel_2025-12-16"
PPU_TOTAL_MINUS_110 = 100.0 / 110.0


# ---------- helpers ----------

def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _get_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[totals] Missing game_date/date column")


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}

    stake = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    roi = profit / stake if stake > 0 else None
    win_rate = float((bets["result"] == "win").mean())

    return {
        "bets": int(len(bets)),
        "stake": stake,
        "profit": profit,
        "roi": roi,
        "win_rate": win_rate,
    }


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser("totals_roi_analysis.py")

    ap.add_argument("--per-game", required=True)
    ap.add_argument("--calibrator", required=True)

    ap.add_argument("--ev", type=float, default=0.04)
    ap.add_argument("--min-abs-residual", type=float, default=2.5)

    ap.add_argument("--max-dispersion", type=float, default=8.0)
    ap.add_argument("--require-dispersion", action="store_true")

    ap.add_argument("--side", default="both", choices=["both", "over", "under"])

    ap.add_argument("--eval-start", required=True)
    ap.add_argument("--eval-end", required=True)

    ap.add_argument("--max-bet-rate", type=float, default=0.15)
    ap.add_argument("--max-profit-abs", type=float, default=10.0)

    ap.add_argument("--out-dir", default="outputs")

    args = ap.parse_args()

    print(f"[totals] version={TOTALS_ROI_VERSION}")
    print(f"[totals] per_game={args.per_game}")
    print(f"[totals] calibrator={args.calibrator}")
    print(f"[totals] ev_base={args.ev}")
    print(f"[totals] min_abs_residual={args.min_abs_residual}")
    print(f"[totals] side_policy={args.side}")

    # ---------- load ----------
    df = pd.read_csv(args.per_game)
    if df.empty:
        raise RuntimeError("[totals] per_game is empty")

    date_col = _get_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    es = pd.to_datetime(args.eval_start)
    ee = pd.to_datetime(args.eval_end)

    df = df[(df[date_col] >= es) & (df[date_col] <= ee)].copy()
    print(f"[totals] eval_window: {es.date()}..{ee.date()} rows={len(df)}")

    if df.empty:
        raise RuntimeError("[totals] No rows in eval window")

    required = [
        "fair_total_model",
        "total_consensus",
        "home_score",
        "away_score",
    ]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"[totals] missing column: {c}")

    # ---------- core math ----------
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["total_residual"] = df["fair_total_model"] - df["total_consensus"]
    df["abs_residual"] = df["total_residual"].abs()

    # residual magnitude gate
    eligible = df["abs_residual"] >= args.min_abs_residual

    # dispersion gate (optional)
    if "total_dispersion" in df.columns:
        df["total_dispersion"] = pd.to_numeric(df["total_dispersion"], errors="coerce")
        disp_ok = df["total_dispersion"] <= args.max_dispersion
        eligible = eligible & disp_ok if args.require_dispersion else eligible
    elif args.require_dispersion:
        raise RuntimeError("[totals] require-dispersion=True but total_dispersion missing")

    df["eligible"] = eligible

    # ---------- calibrator ----------
    cal = joblib.load(args.calibrator)
    iso = cal.get("model")
    if iso is None:
        raise RuntimeError("[totals] invalid calibrator artifact")

    df["p_over"] = iso.predict(df["total_residual"].astype(float))
    df["p_under"] = 1.0 - df["p_over"]

    df["over_ev"] = df["p_over"].apply(expected_value_total)
    df["under_ev"] = df["p_under"].apply(expected_value_total)

    # ---------- EV × residual funnel ----------
    def ev_required(abs_res: float) -> float:
        """
        Larger residual → lower EV requirement
        """
        if abs_res >= 6.0:
            return max(0.03, args.ev - 0.02)
        if abs_res >= 4.0:
            return max(0.035, args.ev - 0.01)
        return args.ev

    def choose_side(r):
        if not r["eligible"]:
            return False, None, None

        req_ev = ev_required(r["abs_residual"])
        oe, ue = r["over_ev"], r["under_ev"]

        if pd.notna(oe) and oe >= req_ev and (pd.isna(ue) or oe > ue):
            return True, "over", oe
        if pd.notna(ue) and ue >= req_ev and (pd.isna(oe) or ue > oe):
            return True, "under", ue
        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0]
    df["bet_side"] = chosen[1]
    df["ev_used"] = chosen[2]

    bets = df[df["bet"]].copy()

    # side policy
    if args.side != "both":
        bets = bets[bets["bet_side"] == args.side]

   total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else len(df)
max_bets = max(1, int(math.floor(args.max_bet_rate * total_games)))

# Rank by EV strength (descending)
bets = bets.sort_values("ev_used", ascending=False)

if len(bets) > max_bets:
    print(
        f"[totals] Bet-rate capped: trimming {len(bets)} → {max_bets} "
        f"(cap={args.max_bet_rate:.2f})"
    )
    bets = bets.head(max_bets)

bet_rate = len(bets) / max(total_games, 1)

        )

    if bets.empty:
        print("[totals] No bets selected")
        return

    # ---------- settlement ----------
    bets["stake"] = 1.0

    def result(r):
        if r["actual_total"] == r["total_consensus"]:
            return "push"
        if r["bet_side"] == "over":
            return "win" if r["actual_total"] > r["total_consensus"] else "loss"
        return "win" if r["actual_total"] < r["total_consensus"] else "loss"

    bets["result"] = bets.apply(result, axis=1)

    bets["profit"] = bets["result"].map(
        {"win": PPU_TOTAL_MINUS_110, "loss": -1.0, "push": 0.0}
    )

    if bets["profit"].abs().max() > args.max_profit_abs:
        raise RuntimeError("[totals] Profit sanity failure")

    # ---------- outputs ----------
    overall = summarize(bets)
    over_sum = summarize(bets[bets["bet_side"] == "over"])
    under_sum = summarize(bets[bets["bet_side"] == "under"])

    print(f"[totals] overall: {overall}")
    print(f"[totals] over_only: {over_sum}")
    print(f"[totals] under_only: {under_sum}")
    print(f"[totals] bet_rate: {bet_rate:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)

    bets.to_csv(os.path.join(args.out_dir, "totals_roi_bets.csv"), index=False)

    metrics = {
        "version": TOTALS_ROI_VERSION,
        "overall": overall,
        "over_only": over_sum,
        "under_only": under_sum,
        "bet_rate": bet_rate,
        "base_ev": args.ev,
        "min_abs_residual": args.min_abs_residual,
        "eval_window": {"start": str(es.date()), "end": str(ee.date())},
        "pricing": {"assumed": "-110", "ppu": PPU_TOTAL_MINUS_110},
    }

    with open(os.path.join(args.out_dir, "totals_roi_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[totals] wrote outputs/totals_roi_metrics.json")
    print("[totals] wrote outputs/totals_roi_bets.csv")


if __name__ == "__main__":
    main()
