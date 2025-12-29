"""
E3 Policy Runner (staking audit + metrics)

Consumes:
- outputs/ats_roi_bets.csv   (selection already locked)
- configs/e3_staking_policy_v1.json

Produces:
- outputs/e3_policy_metrics.json
- outputs/ats_e3_staked_bets.csv (staked bets with reasons)

Deterministic:
- stable sort by (game_date, merge_key)
- sequential state for loss-streak + drawdown brakes
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.staking.e3_staking import load_policy


# ---------------------------
# Odds helpers (American-only)
# ---------------------------

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


# ---------------------------
# Column detection
# ---------------------------

def pick_ev_col(df: pd.DataFrame) -> str:
    candidates = ["ev_units", "ev_used", "away_ev", "metric_used"]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[e3] No EV column found. Tried: {candidates}")


def pick_result_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["result", "outcome", "bet_result"]:
        if c in df.columns:
            return c
    return None


def pick_clv_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["clv_aligned", "clv", "clv_raw_home"]:
        if c in df.columns:
            return c
    return None


def pick_price_col(df: pd.DataFrame) -> Optional[str]:
    # If present in bets, prefer explicit price columns
    for c in ["price_open", "odds_open", "odds_price", "away_price_open", "home_price_open"]:
        if c in df.columns:
            return c
    return None


def infer_profit_per_unit(df: pd.DataFrame) -> pd.Series:
    """
    Determine profit-per-unit (PPU) for wins.
    Preference:
    1) ppu column if present
    2) price column -> ppu via American odds
    3) fallback to fixed -110 (ppu=0.9090909)
    """
    if "ppu" in df.columns:
        s = pd.to_numeric(df["ppu"], errors="coerce")
        if s.notna().any():
            return s

    price_col = pick_price_col(df)
    if price_col is not None:
        s = pd.to_numeric(df[price_col], errors="coerce").apply(clean_american_odds)
        out = s.apply(win_profit_per_unit_american)
        return pd.to_numeric(out, errors="coerce")

    # fixed -110 fallback
    return pd.Series([100.0 / 110.0] * len(df), index=df.index, dtype=float)


# ---------------------------
# E3 sequential sizing simulation
# ---------------------------

@dataclass(frozen=True)
class E3RunConfig:
    ats_bets_csv: str
    policy_json: str
    out_metrics: str
    out_staked_bets: str


def _round_to(x: float, inc: float) -> float:
    return round(x / inc) * inc


def _loss_streak_multiplier(policy: Dict[str, Any], loss_streak: int) -> float:
    damp = 1.0
    if policy["risk_caps"]["loss_streak_dampening"]["enabled"]:
        sched = policy["risk_caps"]["loss_streak_dampening"]["schedule"]
        for row in sched:
            if loss_streak >= int(row["losses"]):
                damp = float(row["multiplier"])
        damp = max(damp, float(policy["risk_caps"]["loss_streak_dampening"]["floor_multiplier"]))
    return float(damp)


def run_e3(cfg: E3RunConfig) -> Dict[str, Any]:
    df = pd.read_csv(cfg.ats_bets_csv)
    if df.empty:
        metrics = {
            "sample_size": {"bets": 0},
            "performance": {"roi": None, "average_clv": None, "clv_positive_rate": None},
            "risk_metrics": {"max_drawdown_units": 0.0},
            "exposure": {"max_daily_stake_u": 0.0, "max_per_bet_stake_u": 0.0},
            "guards": {"daily_cap_violations": 0, "per_bet_cap_violations": 0},
        }
        Path(cfg.out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        df.to_csv(cfg.out_staked_bets, index=False)
        return metrics

    policy = load_policy(cfg.policy_json)

    # stable ordering
    if "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype(str).str[:10]
        df = df.sort_values(["game_date", "merge_key"] if "merge_key" in df.columns else ["game_date"], kind="mergesort")
    else:
        df = df.sort_values(["merge_key"] if "merge_key" in df.columns else list(df.columns)[:1], kind="mergesort")

    ev_col = pick_ev_col(df)
    df[ev_col] = pd.to_numeric(df[ev_col], errors="coerce")
    df = df.dropna(subset=[ev_col]).copy()

    # policy params
    unit_usd = float(policy["unit_convention"]["unit_usd"])
    kelly_base = float(policy["stake_model"]["kelly_fraction"])
    kelly_hard_cap = float(policy["stake_model"]["kelly_fraction_hard_cap"])
    max_stake_u_base = float(policy["stake_model"]["max_stake_u"])
    max_daily_u = float(policy["risk_caps"]["max_daily_stake_u"])
    inc = float(policy["stake_model"]["rounding"]["increment_u"])

    # drawdown brake stages
    stages = policy["risk_caps"]["drawdown_brakes"]["stages"]
    stages = sorted(stages, key=lambda x: float(x["drawdown_u"]))

    # result + clv
    res_col = pick_result_col(df)
    clv_col = pick_clv_col(df)

    # Profit-per-unit for wins
    ppu = infer_profit_per_unit(df)

    # sequential state
    loss_streak = 0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0  # negative or 0
    halt_days_left = 0

    # per-day exposure tracking
    current_day = None
    day_stake_sum = 0.0
    daily_caps_hit = 0

    stake_u_list = []
    reason_list = []
    dd_list = []
    ls_list = []
    kelly_eff_list = []
    max_stake_eff_list = []

    # helper: apply brakes based on current drawdown
    def apply_brakes(drawdown_u: float) -> Tuple[float, float, int]:
        kelly_eff = kelly_base
        max_stake_eff = max_stake_u_base
        halt = 0
        for st in stages:
            if drawdown_u >= float(st["drawdown_u"]):
                if st["action"] == "reduce_kelly":
                    kelly_eff = min(kelly_eff, float(st["new_kelly"]))
                elif st["action"] == "cap_max_stake":
                    max_stake_eff = min(max_stake_eff, float(st["new_max_stake_u"]))
                elif st["action"] == "halt":
                    halt = int(st.get("cooldown_days", 0))
        return kelly_eff, max_stake_eff, halt

    # simulate bet-by-bet
    for i, row in df.iterrows():
        gd = row["game_date"] if "game_date" in df.columns else "unknown"
        if current_day != gd:
            current_day = gd
            day_stake_sum = 0.0

        # current drawdown in units (peak - cum)
        peak = max(peak, cum)
        dd_now = cum - peak  # <= 0
        max_dd = min(max_dd, dd_now)
        drawdown_u = abs(dd_now)

        # halt logic
        if halt_days_left > 0:
            stake_u = 0.0
            reason = "halt"
            halt_days_left = max(0, halt_days_left - (1 if current_day != gd else 0))
        else:
            # brakes
            kelly_eff, max_stake_eff, new_halt = apply_brakes(drawdown_u)
            if new_halt > 0 and drawdown_u >= float([s for s in stages if s["action"] == "halt"][-1]["drawdown_u"]):
                stake_u = 0.0
                reason = "halt"
                halt_days_left = new_halt
            else:
                # loss-streak dampening
                damp = _loss_streak_multiplier(policy, loss_streak)

                ev = float(row[ev_col])
                base_u = max(0.0, kelly_eff * ev)
                base_u = min(base_u, kelly_hard_cap)
                stake_u = min(max_stake_eff, base_u * damp)
                stake_u = max(0.0, _round_to(stake_u, inc))

                # daily cap
                if (day_stake_sum + stake_u) > max_daily_u:
                    stake_u = 0.0
                    reason = "daily_cap"
                    daily_caps_hit += 1
                else:
                    reason = "accepted"
                    day_stake_sum += stake_u

        # apply result to PnL
        profit = 0.0
        if stake_u > 0 and res_col is not None:
            r = str(row[res_col]).lower().strip()
            p = ppu.loc[i] if i in ppu.index else np.nan
            p = float(p) if (p is not None and np.isfinite(p)) else (100.0 / 110.0)

            if r == "win":
                profit = stake_u * p
                loss_streak = 0
            elif r == "push":
                profit = 0.0
                # loss_streak unchanged
            else:
                profit = -stake_u
                loss_streak += 1

        cum += profit
        # update dd after profit
        peak = max(peak, cum)
        dd_now = cum - peak
        max_dd = min(max_dd, dd_now)

        stake_u_list.append(float(stake_u))
        reason_list.append(reason)
        dd_list.append(float(dd_now))
        ls_list.append(int(loss_streak))

        # record effective params (for audit)
        k_eff, m_eff, _ = apply_brakes(abs(dd_now))
        kelly_eff_list.append(float(k_eff))
        max_stake_eff_list.append(float(m_eff))

    out = df.copy()
    out["stake_u"] = stake_u_list
    out["stake_usd"] = out["stake_u"] * float(unit_usd)
    out["e3_reason"] = reason_list
    out["loss_streak"] = ls_list
    out["drawdown_units"] = dd_list
    out["kelly_effective"] = kelly_eff_list
    out["max_stake_effective_u"] = max_stake_eff_list

    Path(os.path.dirname(cfg.out_staked_bets) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_staked_bets, index=False)

    # Aggregate E3 metrics (on bets with stake>0)
    bet_mask = out["stake_u"] > 0
    stake_sum = float(out.loc[bet_mask, "stake_u"].sum())
    profit_sum = float((out.loc[bet_mask, "stake_u"] * 0.0).sum())  # placeholder

    # Compute realized profit from result if present
    if res_col is not None:
        # reuse ppu
        realized = []
        for idx, r in out.loc[bet_mask].iterrows():
            rr = str(r[res_col]).lower().strip()
            p = ppu.loc[idx] if idx in ppu.index else np.nan
            p = float(p) if (p is not None and np.isfinite(p)) else (100.0 / 110.0)
            su = float(r["stake_u"])
            if rr == "win":
                realized.append(su * p)
            elif rr == "push":
                realized.append(0.0)
            else:
                realized.append(-su)
        profit_sum = float(np.sum(realized))

    roi = (profit_sum / stake_sum) if stake_sum > 0 else None

    # CLV summary if available
    avg_clv = None
    clv_pos_rate = None
    if clv_col is not None:
        clv_s = pd.to_numeric(out.loc[bet_mask, clv_col], errors="coerce")
        if clv_s.notna().any():
            avg_clv = float(clv_s.mean())
            clv_pos_rate = float((clv_s > 0).mean())

    # Exposure checks
    max_daily = 0.0
    if "game_date" in out.columns:
        max_daily = float(out.groupby("game_date")["stake_u"].sum().max())

    max_per_bet = float(out["stake_u"].max())

    # Guard violations
    per_bet_viol = int((out["stake_u"] > float(policy["stake_model"]["max_stake_u"]) + 1e-9).sum())
    daily_viol = int((out.groupby("game_date")["stake_u"].sum() > float(policy["risk_caps"]["max_daily_stake_u"]) + 1e-9).sum()) if "game_date" in out.columns else 0

    metrics = {
        "sample_size": {"bets": int(bet_mask.sum()), "rows": int(len(out))},
        "performance": {"roi": roi, "profit_units": profit_sum, "stake_units": stake_sum, "average_clv": avg_clv, "clv_positive_rate": clv_pos_rate},
        "risk_metrics": {"max_drawdown_units": float(max_dd)},
        "exposure": {"max_daily_stake_u": max_daily, "max_per_bet_stake_u": max_per_bet, "daily_caps_hit": int(daily_caps_hit)},
        "guards": {"daily_cap_violations": daily_viol, "per_bet_cap_violations": per_bet_viol},
        "policy": {"path": cfg.policy_json, "name": policy.get("policy_name"), "version": policy.get("version")},
    }

    Path(os.path.dirname(cfg.out_metrics) or ".").mkdir(parents=True, exist_ok=True)
    Path(cfg.out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser("e3_policy_runner.py")
    ap.add_argument("--ats-bets", required=True, help="Path to outputs/ats_roi_bets.csv")
    ap.add_argument("--policy", required=True, help="Path to configs/e3_staking_policy_v1.json")
    ap.add_argument("--out-metrics", default="outputs/e3_policy_metrics.json")
    ap.add_argument("--out-bets", default="outputs/ats_e3_staked_bets.csv")
    args = ap.parse_args()

    cfg = E3RunConfig(
        ats_bets_csv=str(args.ats_bets),
        policy_json=str(args.policy),
        out_metrics=str(args.out_metrics),
        out_staked_bets=str(args.out_bets),
    )

    m = run_e3(cfg)
    print(f"[e3] wrote: {cfg.out_metrics}")
    print(f"[e3] wrote: {cfg.out_staked_bets}")
    print(f"[e3] bets={m['sample_size']['bets']} roi={m['performance']['roi']} max_dd={m['risk_metrics']['max_drawdown_units']}")
    print(f"[e3] exposure max_daily={m['exposure']['max_daily_stake_u']} max_per_bet={m['exposure']['max_per_bet_stake_u']}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", ".")
    main()
