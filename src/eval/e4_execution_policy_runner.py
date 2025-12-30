"""
E4.2 Execution Timing Policy Runner (ATS v1 away-only)

Goal
----
Turn the time-of-day diagnostics into a *single executable decision* per game,
using conservative, deterministic gates.

Inputs
------
1) outputs/ats_time_of_day_diagnostics.csv
   - produced by src.eval.time_of_day_ev_diagnostics
   - contains open/close consensus + dispersion, calibrated probs, EVs, bet flags

2) outputs/backtest_joined_market.csv  (or backtest_joined.csv)
   - contains final scores + home/away teams + merge_key (or enough to build it)

3) Optional per-book snapshot CSVs in data/_snapshots for book coverage
   - open_YYYYMMDD.csv and close_YYYYMMDD.csv
   - columns: merge_key, book, spread_home_point (normalized wide format)

Outputs
-------
- outputs/e4_execution_policy_bets.csv
- outputs/e4_execution_policy_metrics.json

Policy (defaults; tune later but keep deterministic)
----------------------------------------------------
OPEN execute if:
  - ev_away_open >= 0.035
  - open_dispersion <= 1.75
  - book_count_open >= 4 (if available; else ignored)

CLOSE execute if NOT open_execute and:
  - ev_away_close >= 0.040
  - close_dispersion <= 1.50
  - (ev_away_open is NaN OR ev_away_open >= 0.0)  # avoid reversal traps
  - book_count_close >= 4 (if available; else ignored)

Settlement
----------
Uses executed home spread:
  - OPEN: open_consensus
  - CLOSE: close_consensus
Profit at -110:
  win: +100/110
  loss: -1
  push: 0

Notes
-----
- This runner does NOT change the model or selector math.
- It only converts diagnostics into a timing+execution decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


PPU_ATS_MINUS_110 = 100.0 / 110.0


def _norm_key(s: object) -> str:
    return str(s).strip().lower()


def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return out
    for c in ("date", "gamedate"):
        if c in out.columns:
            out["game_date"] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
            return out
    raise RuntimeError("Missing game_date/date column")


def _load_book_counts(snapshot_dir: Path, kind: str) -> pd.DataFrame:
    """
    Compute per (game_date, merge_key) count of distinct books posting spreads.
    Requires normalized per-book CSV snapshots: {kind}_YYYYMMDD.csv with columns:
      merge_key, book, spread_home_point
    """
    kind = kind.lower().strip()
    files = sorted(snapshot_dir.glob(f"{kind}_*.csv"))
    if not files:
        return pd.DataFrame(columns=["game_date", "merge_key", f"book_count_{kind}"])

    frames = []
    for p in files:
        name = p.name
        # kind_YYYYMMDD.csv
        try:
            ymd = name.split("_", 1)[1].split(".", 1)[0]
            if len(ymd) != 8:
                continue
            game_date = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        except Exception:
            continue

        df = pd.read_csv(p)
        if df.empty:
            continue
        if "merge_key" not in df.columns:
            continue

        # Detect spread column
        spread_col = None
        for c in ("spread_home_point", "spread_home", "spread"):
            if c in df.columns:
                spread_col = c
                break
        if spread_col is None:
            continue

        book_col = "book" if "book" in df.columns else None
        if book_col is None:
            # if no book column, treat each row as one book (worst-case 1)
            tmp = df[["merge_key", spread_col]].copy()
            tmp["book"] = "unknown"
            book_col = "book"
            df = tmp

        df = df.copy()
        df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
        df[spread_col] = pd.to_numeric(df[spread_col], errors="coerce")
        df[book_col] = df[book_col].astype(str)

        usable = df.dropna(subset=["merge_key", spread_col])
        if usable.empty:
            continue

        agg = (
            usable.groupby("merge_key")[book_col]
            .nunique()
            .rename(f"book_count_{kind}")
            .reset_index()
        )
        agg["game_date"] = game_date
        frames.append(agg[["game_date", "merge_key", f"book_count_{kind}"]])

    if not frames:
        return pd.DataFrame(columns=["game_date", "merge_key", f"book_count_{kind}"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["game_date", "merge_key"]).drop_duplicates(["game_date", "merge_key"], keep="last")
    return out


def _settle_away(home_score: float, away_score: float, home_spread: float) -> str:
    adj_home = home_score + home_spread
    if adj_home == away_score:
        return "push"
    return "win" if adj_home < away_score else "loss"  # away covers if adjusted home loses


def run_policy(
    *,
    diagnostics_csv: str,
    backtest_csv: str,
    snapshot_dir: Optional[str] = None,
    out_csv: str = "outputs/e4_execution_policy_bets.csv",
    out_json: str = "outputs/e4_execution_policy_metrics.json",
    # thresholds
    ev_open: float = 0.035,
    disp_open: float = 1.75,
    ev_close: float = 0.040,
    disp_close: float = 1.50,
    min_books: int = 4,
) -> Tuple[pd.DataFrame, Dict]:

    diag = pd.read_csv(diagnostics_csv)
    if diag.empty:
        raise RuntimeError("diagnostics_csv is empty")

    back = pd.read_csv(backtest_csv)
    if back.empty:
        raise RuntimeError("backtest_csv is empty")

    diag = _ensure_game_date(diag)
    back = _ensure_game_date(back)

    # normalize merge_key
    if "merge_key" not in diag.columns:
        raise RuntimeError("diagnostics missing merge_key")
    if "merge_key" not in back.columns:
        raise RuntimeError("backtest missing merge_key (expected from build_ats_roi_input/backtest pipelines)")

    diag["merge_key"] = diag["merge_key"].astype(str).str.strip().str.lower()
    back["merge_key"] = back["merge_key"].astype(str).str.strip().str.lower()

    # optional book counts
    if snapshot_dir:
        sdir = Path(snapshot_dir)
        bc_open = _load_book_counts(sdir, "open")
        bc_close = _load_book_counts(sdir, "close")
        diag = diag.merge(bc_open, on=["game_date", "merge_key"], how="left").merge(
            bc_close, on=["game_date", "merge_key"], how="left"
        )
    else:
        diag["book_count_open"] = np.nan
        diag["book_count_close"] = np.nan

    # attach scores + teams
    keep_back = []
    for c in ("home_team", "away_team", "home_score", "away_score"):
        if c in back.columns:
            keep_back.append(c)
    if "home_score" not in keep_back or "away_score" not in keep_back:
        # fallbacks from earlier pipelines
        for pair in (("home_final_score", "away_final_score"), ("home_pts", "away_pts")):
            if pair[0] in back.columns and pair[1] in back.columns:
                back = back.rename(columns={pair[0]: "home_score", pair[1]: "away_score"})
                keep_back = [c for c in keep_back if c not in pair] + ["home_score", "away_score"]
                break
    keep_back = list(dict.fromkeys(keep_back))  # de-dupe

    merged = diag.merge(
        back[["game_date", "merge_key"] + keep_back],
        on=["game_date", "merge_key"],
        how="left",
    )

    # required diagnostic fields
    required = {"open_consensus", "close_consensus", "open_dispersion", "close_dispersion", "ev_away_open", "ev_away_close"}
    missing = required - set(merged.columns)
    if missing:
        raise RuntimeError(f"Missing required columns after merge: {missing}")

    # numeric coercions
    merged["open_consensus"] = pd.to_numeric(merged["open_consensus"], errors="coerce")
    merged["close_consensus"] = pd.to_numeric(merged["close_consensus"], errors="coerce")
    merged["open_dispersion"] = pd.to_numeric(merged["open_dispersion"], errors="coerce")
    merged["close_dispersion"] = pd.to_numeric(merged["close_dispersion"], errors="coerce")
    merged["ev_away_open"] = pd.to_numeric(merged["ev_away_open"], errors="coerce")
    merged["ev_away_close"] = pd.to_numeric(merged["ev_away_close"], errors="coerce")

    # book gates
    def _book_ok(x) -> bool:
        if pd.isna(x):
            return True  # if missing coverage data, don't fail (diagnostic-only)
        return int(x) >= int(min_books)

    book_ok_open = merged["book_count_open"].apply(_book_ok)
    book_ok_close = merged["book_count_close"].apply(_book_ok)

    # OPEN execute
    open_ok = (
        (merged["ev_away_open"] >= float(ev_open))
        & (merged["open_dispersion"] <= float(disp_open))
        & book_ok_open
    )

    # CLOSE execute (fallback)
    close_ok = (
        (~open_ok)
        & (merged["ev_away_close"] >= float(ev_close))
        & (merged["close_dispersion"] <= float(disp_close))
        & book_ok_close
        & (merged["ev_away_open"].isna() | (merged["ev_away_open"] >= 0.0))
    )

    merged["execute_window"] = np.where(open_ok, "OPEN", np.where(close_ok, "CLOSE", "NO_BET"))
    merged["executed_home_spread"] = np.where(
        merged["execute_window"] == "OPEN",
        merged["open_consensus"],
        np.where(merged["execute_window"] == "CLOSE", merged["close_consensus"], np.nan),
    )

    # CLV relative to close consensus (for reporting)
    merged["clv_vs_close"] = merged["close_consensus"] - merged["executed_home_spread"]

    # settle + profit
    hs = pd.to_numeric(merged.get("home_score"), errors="coerce")
    aw = pd.to_numeric(merged.get("away_score"), errors="coerce")
    line = pd.to_numeric(merged["executed_home_spread"], errors="coerce")

    results = []
    profits = []
    for h, a, l, w in zip(hs, aw, line, merged["execute_window"]):
        if w == "NO_BET" or pd.isna(h) or pd.isna(a) or pd.isna(l):
            results.append("no_bet")
            profits.append(0.0)
            continue
        r = _settle_away(float(h), float(a), float(l))
        results.append(r)
        if r == "win":
            profits.append(PPU_ATS_MINUS_110)
        elif r == "loss":
            profits.append(-1.0)
        else:
            profits.append(0.0)

    merged["result"] = results
    merged["profit_u"] = profits

    bets = merged[merged["execute_window"] != "NO_BET"].copy()
    bets = bets.sort_values(["game_date", "merge_key"]).reset_index(drop=True)

    # summary metrics
    pnl = bets["profit_u"].to_numpy(dtype=float)
    cum = np.cumsum(pnl) if len(pnl) else np.array([0.0])
    running_max = np.maximum.accumulate(cum) if len(pnl) else np.array([0.0])
    dd = cum - running_max
    max_dd = float(np.min(dd)) if len(pnl) else 0.0

    n = int(len(bets))
    roi = float(cum[-1] / n) if n else 0.0
    win_rate = float((bets["result"] == "win").mean()) if n else None
    clv = float(np.nanmean(bets["clv_vs_close"])) if n else None

    summary = {
        "counts": {
            "rows": int(len(merged)),
            "bets": n,
            "open_bets": int((bets["execute_window"] == "OPEN").sum()),
            "close_bets": int((bets["execute_window"] == "CLOSE").sum()),
        },
        "thresholds": {
            "ev_open": ev_open,
            "disp_open": disp_open,
            "ev_close": ev_close,
            "disp_close": disp_close,
            "min_books": min_books,
        },
        "metrics": {
            "roi_per_bet_u": roi,
            "profit_u": float(cum[-1]) if n else 0.0,
            "win_rate": win_rate,
            "avg_clv_vs_close": clv,
            "max_drawdown_u": max_dd,
        },
        "artifacts": {"bets_csv": out_csv, "metrics_json": out_json},
    }

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(out_csv, index=False)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return bets, summary


def main() -> None:
    ap = argparse.ArgumentParser("e4_execution_policy_runner")
    ap.add_argument("--diagnostics", default="outputs/ats_time_of_day_diagnostics.csv")
    ap.add_argument("--backtest", default="outputs/backtest_joined_market.csv")
    ap.add_argument("--snapshot-dir", default=None, help="data/_snapshots (optional, enables book coverage)")
    ap.add_argument("--out-csv", default="outputs/e4_execution_policy_bets.csv")
    ap.add_argument("--out-json", default="outputs/e4_execution_policy_metrics.json")

    ap.add_argument("--ev-open", type=float, default=0.035)
    ap.add_argument("--disp-open", type=float, default=1.75)
    ap.add_argument("--ev-close", type=float, default=0.040)
    ap.add_argument("--disp-close", type=float, default=1.50)
    ap.add_argument("--min-books", type=int, default=4)

    args = ap.parse_args()

    run_policy(
        diagnostics_csv=args.diagnostics,
        backtest_csv=args.backtest,
        snapshot_dir=args.snapshot_dir,
        out_csv=args.out_csv,
        out_json=args.out_json,
        ev_open=float(args.ev_open),
        disp_open=float(args.disp_open),
        ev_close=float(args.ev_close),
        disp_close=float(args.disp_close),
        min_books=int(args.min_books),
    )
    print(f"[e4] wrote {args.out_csv}")
    print(f"[e4] wrote {args.out_json}")


if __name__ == "__main__":
    main()
