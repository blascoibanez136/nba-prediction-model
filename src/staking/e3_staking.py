from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

def _round_to(x: float, inc: float) -> float:
    return round(x / inc) * inc

def load_policy(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_e3_staking(
    ats_bets_csv: str,
    policy_json: str,
    *,
    loss_streak: int = 0,
    drawdown_u: float = 0.0
) -> pd.DataFrame:
    """
    Input: outputs/ats_roi_bets.csv (selection already locked)
    Output: DataFrame with stake_u, stake_usd, and audit flags.
    """
    df = pd.read_csv(ats_bets_csv)
    if df.empty:
        df["stake_u"] = 0.0
        df["stake_usd"] = 0.0
        df["e3_reason"] = "no_bets"
        return df

    policy = load_policy(policy_json)

    unit_usd = policy["unit_convention"]["unit_usd"]
    kelly = policy["stake_model"]["kelly_fraction"]
    kelly_cap = policy["stake_model"]["kelly_fraction_hard_cap"]
    max_stake_u = policy["stake_model"]["max_stake_u"]
    max_daily_u = policy["risk_caps"]["max_daily_stake_u"]
    inc = policy["stake_model"]["rounding"]["increment_u"]

    # Drawdown brakes
    for stage in policy["risk_caps"]["drawdown_brakes"]["stages"]:
        if drawdown_u >= stage["drawdown_u"]:
            if stage["action"] == "reduce_kelly":
                kelly = min(kelly, stage["new_kelly"])
            if stage["action"] == "cap_max_stake":
                max_stake_u = min(max_stake_u, stage["new_max_stake_u"])
            if stage["action"] == "halt":
                df["stake_u"] = 0.0
                df["stake_usd"] = 0.0
                df["e3_reason"] = "halt"
                return df

    # Loss-streak dampening
    damp = 1.0
    if policy["risk_caps"]["loss_streak_dampening"]["enabled"]:
        sched = policy["risk_caps"]["loss_streak_dampening"]["schedule"]
        for row in sched:
            if loss_streak >= row["losses"]:
                damp = row["multiplier"]
        damp = max(damp, policy["risk_caps"]["loss_streak_dampening"]["floor_multiplier"])

    # Expect EV per 1u present as `ev_units`
    if "ev_units" not in df.columns:
        raise RuntimeError("ats_roi_bets.csv missing ev_units (required for E3 sizing).")

    # Kelly-lite sizing per bet
    df = df.copy()
    raw_u = kelly * df["ev_units"].clip(lower=0)
    raw_u = raw_u.clip(upper=kelly_cap)
    stake_u = (raw_u * damp).clip(upper=max_stake_u)
    stake_u = stake_u.apply(lambda x: _round_to(x, inc))

    df["stake_u"] = stake_u
    df["stake_usd"] = df["stake_u"] * unit_usd

    # Order by EV, then CLV if present
    order = policy["ordering"]
    if "clv" in df.columns:
        df = df.sort_values(
            by=[order["primary"].replace("_desc",""), order["secondary"].replace("_desc","")],
            ascending=[False, False]
        )
    else:
        df = df.sort_values(by=order["primary"].replace("_desc",""), ascending=False)

    # Daily cap trim
    cum = df["stake_u"].cumsum()
    mask = cum <= max_daily_u
    df.loc[~mask, ["stake_u","stake_usd"]] = 0.0
    df["e3_reason"] = np.where(mask, "accepted", "daily_cap")

    return df
