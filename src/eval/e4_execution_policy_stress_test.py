"""E4.3 Stress Test on Execution-Policy Bets
=========================================

This script stress-tests the execution-aware bet list produced by E4.2:
    outputs/e4_execution_policy_bets.csv

It differs from E4.1: it operates on a *bet list* (already OPEN/CLOSE decided),
not a per-game open/close consensus table.

Required columns (auto-detected with fallbacks)
----------------------------------------------
- execute_window            (OPEN/CLOSE)
- executed_home_spread      (the home spread used for settlement)
- home_score, away_score    (final scores) or common aliases

Optional columns (improves WORST_BOOK proxy)
--------------------------------------------
- open_consensus, close_consensus
- open_dispersion, close_dispersion

Modes
-----
BASELINE   : use executed_home_spread as-is
PLUS_0.25  : executed_home_spread + 0.25  (worse for AWAY)
PLUS_0.50  : executed_home_spread + 0.50  (worse for AWAY)
WORST_BOOK : proxy worst fill using (consensus + |dispersion|) for the relevant window

Profit model: -110, 1u stake (win +100/110, loss -1, push 0)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

PPU_ATS_MINUS_110 = 100.0 / 110.0


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _settle_away(hs: float, aw: float, home_spread: float) -> str:
    adj_home = hs + home_spread
    if adj_home == aw:
        return "push"
    return "win" if adj_home < aw else "loss"


def _profit(result: str) -> float:
    r = str(result).lower()
    if r == "win":
        return PPU_ATS_MINUS_110
    if r == "loss":
        return -1.0
    return 0.0


def _worst_book_proxy(row: pd.Series) -> float:
    w = str(row.get("execute_window", "")).upper().strip()
    if w == "OPEN":
        oc = row.get("open_consensus", np.nan)
        od = row.get("open_dispersion", np.nan)
        if pd.notna(oc) and pd.notna(od):
            return float(oc) + abs(float(od))
    if w == "CLOSE":
        cc = row.get("close_consensus", np.nan)
        cd = row.get("close_dispersion", np.nan)
        if pd.notna(cc) and pd.notna(cd):
            return float(cc) + abs(float(cd))
    return float(row.get("executed_home_spread", np.nan))


@dataclass(frozen=True)
class StressRow:
    execution_mode: str
    bets: int
    roi_per_bet_u: float
    profit_u: float
    win_rate: float
    max_drawdown_u: float


def run(bets_csv: str) -> pd.DataFrame:
    df = pd.read_csv(bets_csv)
    if df.empty:
        raise RuntimeError("Input bet list is empty")

    exec_col = _first_col(df, ["execute_window"])
    line_col = _first_col(df, ["executed_home_spread"])
    hs_col = _first_col(df, ["home_score", "home_final_score", "home_pts"])
    aw_col = _first_col(df, ["away_score", "away_final_score", "away_pts"])

    missing = [n for n, c in [("execute_window", exec_col), ("executed_home_spread", line_col), ("home_score", hs_col), ("away_score", aw_col)] if c is None]
    if missing:
        raise RuntimeError(f"Missing required columns in bet list: {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df[exec_col] = df[exec_col].astype(str)
    df["execute_window"] = df[exec_col].str.upper().str.strip()
    df["executed_home_spread"] = _to_num(df[line_col])
    df["home_score"] = _to_num(df[hs_col])
    df["away_score"] = _to_num(df[aw_col])

    for c in ["open_consensus", "close_consensus", "open_dispersion", "close_dispersion"]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    modes = ["BASELINE", "PLUS_0.25", "PLUS_0.50", "WORST_BOOK"]
    rows: list[StressRow] = []

    for mode in modes:
        if mode == "BASELINE":
            line = df["executed_home_spread"].astype(float)
        elif mode == "PLUS_0.25":
            line = df["executed_home_spread"].astype(float) + 0.25
        elif mode == "PLUS_0.50":
            line = df["executed_home_spread"].astype(float) + 0.50
        else:
            line = df.apply(_worst_book_proxy, axis=1)

        results = [
            _settle_away(float(h), float(a), float(l))
            if pd.notna(h) and pd.notna(a) and pd.notna(l)
            else "push"
            for h, a, l in zip(df["home_score"], df["away_score"], line)
        ]
        profits = np.array([_profit(r) for r in results], dtype=float)

        n = int(len(profits))
        cum = np.cumsum(profits) if n else np.array([0.0])
        roi = float(cum[-1] / n) if n else 0.0
        win_rate = float(np.mean(np.array(results) == "win")) if n else 0.0

        running_max = np.maximum.accumulate(cum) if n else np.array([0.0])
        dd = cum - running_max
        max_dd = float(np.min(dd)) if n else 0.0

        rows.append(StressRow(mode, n, roi, float(cum[-1]) if n else 0.0, win_rate, max_dd))

    return pd.DataFrame([r.__dict__ for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser("e4_execution_policy_stress_test")
    ap.add_argument("bets_csv", help="Path to outputs/e4_execution_policy_bets.csv")
    ap.add_argument("--output", default="outputs/e4_execution_policy_stress.csv")
    args = ap.parse_args()

    res = run(args.bets_csv)
    res.to_csv(args.output, index=False)
    print(res.to_string(index=False))
    print(f"[e4.3] wrote {args.output}")


if __name__ == "__main__":
    main()
