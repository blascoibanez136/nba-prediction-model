"""
Backtest predictions against historical results.

Commit-2 intent:
- Load prediction CSVs produced by `src.eval.historical_prediction_runner`.
- Join to a historical results CSV and compute basic scoring/edge metrics.

Commit-3 additive:
- Produce audit-only calibration metrics (bucketed reliability, Brier, logloss)
  without modifying existing outputs or behavior.

Outputs:
- outputs/backtest_metrics.csv
- outputs/backtest_joined.csv
- outputs/backtest_join_audit.json
- outputs/backtest_calibration.csv            (NEW, Commit-3)
- outputs/audits/backtest_calibration.json    (NEW, Commit-3)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Schema inference helpers
# -----------------------------

_SCORE_CANDIDATES_HOME = [
    "home_score",
    "home_points",
    "home_pts",
    "pts_home",
    "home_team_score",
    "home_team_points",
    "home_final",
    "home_final_score",
    "home_score_final",
    "homeTeamScore",
]
_SCORE_CANDIDATES_AWAY = [
    "away_score",
    "away_points",
    "away_pts",
    "pts_away",
    "away_team_score",
    "away_team_points",
    "away_final",
    "away_final_score",
    "away_score_final",
    "awayTeamScore",
]

_DATE_CANDIDATES = [
    "date",
    "game_date",
    "gameDate",
    "start_date",
    "startDate",
    "commence_time",
    "commenceTime",
]

_GAME_ID_CANDIDATES = [
    "game_id",
    "id",
    "gameId",
    "GAME_ID",
]

_HOME_TEAM_CANDIDATES = [
    "home_team",
    "home",
    "home_team_name",
    "homeTeam",
    "HOME_TEAM",
]

_AWAY_TEAM_CANDIDATES = [
    "away_team",
    "away",
    "away_team_name",
    "awayTeam",
    "AWAY_TEAM",
]


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _ensure_date_col(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    df = df.copy()
    date_col = _first_present(df, _DATE_CANDIDATES)
    if not date_col:
        raise KeyError(f"[{label}] Could not find a date column.")
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def _ensure_score_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h = _first_present(df, _SCORE_CANDIDATES_HOME)
    a = _first_present(df, _SCORE_CANDIDATES_AWAY)

    if h and h != "home_score":
        df = df.rename(columns={h: "home_score"})
    if a and a != "away_score":
        df = df.rename(columns={a: "away_score"})

    if "home_score" not in df.columns or "away_score" not in df.columns:
        raise KeyError("Could not find home_score / away_score columns")

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df


def _ensure_join_keys(df: pd.DataFrame, *, label: str) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()

    gid = _first_present(df, _GAME_ID_CANDIDATES)
    if gid:
        if gid != "game_id":
            df = df.rename(columns={gid: "game_id"})
        return df, ["game_id"]

    home = _first_present(df, _HOME_TEAM_CANDIDATES)
    away = _first_present(df, _AWAY_TEAM_CANDIDATES)
    if not home or not away:
        raise KeyError(f"[{label}] Could not find team columns")

    if home != "home_team":
        df = df.rename(columns={home: "home_team"})
    if away != "away_team":
        df = df.rename(columns={away: "away_team"})

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    df = _ensure_date_col(df, label=label)
    return df, ["date", "home_team", "away_team"]


# -----------------------------
# Core loading
# -----------------------------

def _load_predictions(pred_dir: str, pattern: str) -> pd.DataFrame:
    paths = sorted(Path(pred_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError("No prediction files found")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["_pred_file"] = p.name
        frames.append(df)

    preds = pd.concat(frames, ignore_index=True)
    preds = _ensure_date_col(preds, label="preds")
    preds, join_keys = _ensure_join_keys(preds, label="preds")
    preds["_join_key_mode"] = "|".join(join_keys)
    return preds


def _load_history(history_csv: str) -> pd.DataFrame:
    hist = pd.read_csv(history_csv)
    hist = _ensure_date_col(hist, label="history")
    hist, join_keys = _ensure_join_keys(hist, label="history")
    hist["_join_key_mode"] = "|".join(join_keys)
    hist = _ensure_score_cols(hist)
    return hist


# -----------------------------
# Metrics
# -----------------------------

def _compute_metrics(joined: pd.DataFrame, prob_col: str, spread_col: str, total_col: str) -> pd.DataFrame:
    out = joined.copy()

    out["home_margin"] = out["home_score"] - out["away_score"]
    out["total_points"] = out["home_score"] + out["away_score"]
    out["home_win"] = (out["home_margin"] > 0).astype(int)

    if prob_col in out.columns:
        p = pd.to_numeric(out[prob_col], errors="coerce").clip(0, 1)
        out["logloss_component"] = -(
            out["home_win"] * np.log(np.clip(p, 1e-9, 1 - 1e-9))
            + (1 - out["home_win"]) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9))
        )
        out["brier_component"] = (p - out["home_win"]) ** 2

    if spread_col in out.columns:
        s = pd.to_numeric(out[spread_col], errors="coerce")
        out["spread_error"] = (s - out["home_margin"]).abs()

    if total_col in out.columns:
        t = pd.to_numeric(out[total_col], errors="coerce")
        out["total_error"] = (t - out["total_points"]).abs()

    return out


def _summarize(joined: pd.DataFrame) -> Dict[str, float]:
    def _mean(col: str) -> float:
        return float(pd.to_numeric(joined.get(col), errors="coerce").mean())

    return {
        "n_rows": float(len(joined)),
        "n_complete_scores": float(joined[["home_score", "away_score"]].dropna().shape[0]),
        "mean_logloss": _mean("logloss_component"),
        "mean_brier": _mean("brier_component"),
        "mae_spread": _mean("spread_error"),
        "mae_total": _mean("total_error"),
    }


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--pattern", default="predictions_*.csv")
    p.add_argument("--prob-col", default="home_win_prob")
    p.add_argument("--spread-col", default="fair_spread")
    p.add_argument("--total-col", default="fair_total")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    out_dir = Path(args.out_dir or args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(args.pred_dir, args.pattern)
    hist = _load_history(args.history)

    preds = preds[(preds["date"] >= start) & (preds["date"] <= end)]
    hist = hist[(hist["date"] >= start) & (hist["date"] <= end)]

    join_keys = ["game_id"] if "game_id" in preds.columns and "game_id" in hist.columns else ["date", "home_team", "away_team"]
    joined = preds.merge(hist, on=join_keys, how="inner")

    audit = {
        "pred_rows": int(len(preds)),
        "hist_rows": int(len(hist)),
        "joined_rows": int(len(joined)),
        "join_keys": join_keys,
    }
    (out_dir / "backtest_join_audit.json").write_text(json.dumps(audit, indent=2))

    if joined.empty:
        raise RuntimeError("Backtest join produced 0 rows")

    joined_scored = _compute_metrics(joined, args.prob_col, args.spread_col, args.total_col)
    joined_scored.to_csv(out_dir / "backtest_joined.csv", index=False)
    pd.DataFrame([_summarize(joined_scored)]).to_csv(out_dir / "backtest_metrics.csv", index=False)

    # ---------------------------------------------------------
    # Commit-3 ADDITIVE: calibration metrics (audit-only)
    # ---------------------------------------------------------
    try:
        from src.eval.backtest_calibration import calibration_table

        cal_df, cal_summary = calibration_table(joined_scored)

        cal_df.to_csv(out_dir / "backtest_calibration.csv", index=False)

        audits_dir = out_dir / "audits"
        audits_dir.mkdir(exist_ok=True)

        (audits_dir / "backtest_calibration.json").write_text(
            json.dumps(
                {
                    "n": cal_summary.n,
                    "n_used": cal_summary.n_used,
                    "n_dropped": cal_summary.n_dropped,
                    "brier": cal_summary.brier,
                    "logloss": cal_summary.logloss,
                    "dropped_reasons": cal_summary.dropped_reasons,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as e:
        print(f"[backtest] calibration metrics failed (non-fatal): {e!r}")


if __name__ == "__main__":
    main()

