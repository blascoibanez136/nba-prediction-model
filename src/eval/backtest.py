"""Backtest predictions against historical results.

Commit-2 intent:
- Load prediction CSVs produced by `src.eval.historical_prediction_runner`.
- Join to a historical results CSV and compute basic scoring/edge metrics.

Robustness fixes:
- Historical results schemas vary. We infer/rename common score and key columns
  (home_score/away_score, date, teams, game_id).
- Prediction filename pattern defaults to `predictions_*.csv` (matches runner).

Outputs:
- outputs/backtest_metrics.csv
- outputs/backtest_joined.csv
- outputs/backtest_join_audit.json

Note: This module does not assume the presence of market columns; it will
compute what it can from whatever columns exist.
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
    # also allow case-insensitive matches
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _ensure_date_col(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    df = df.copy()
    date_col = _first_present(df, _DATE_CANDIDATES)
    if not date_col:
        raise KeyError(
            f"[{label}] Could not find a date column. Tried: {_DATE_CANDIDATES}. Available: {list(df.columns)}"
        )
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

    # Some historical files store score as a string "102-98".
    if "home_score" not in df.columns or "away_score" not in df.columns:
        # try common combined formats
        for combo in ["score", "final_score", "final", "result"]:
            if combo in df.columns:
                parts = df[combo].astype(str).str.replace(" ", "", regex=False).str.split("-", n=1, expand=True)
                if parts.shape[1] == 2:
                    df["home_score"] = pd.to_numeric(parts[0], errors="coerce")
                    df["away_score"] = pd.to_numeric(parts[1], errors="coerce")
                    break

    if "home_score" not in df.columns or "away_score" not in df.columns:
        raise KeyError(
            "Could not find/derive home_score and away_score in history results. "
            f"Columns: {list(df.columns)}"
        )

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df


def _ensure_join_keys(df: pd.DataFrame, *, label: str) -> Tuple[pd.DataFrame, List[str]]:
    """Return df and a list of join keys to use.

    Preferred join key order:
    1) game_id if available
    2) (date, home_team, away_team)
    """
    df = df.copy()

    gid = _first_present(df, _GAME_ID_CANDIDATES)
    if gid:
        if gid != "game_id":
            df = df.rename(columns={gid: "game_id"})
        return df, ["game_id"]

    home = _first_present(df, _HOME_TEAM_CANDIDATES)
    away = _first_present(df, _AWAY_TEAM_CANDIDATES)
    if not home or not away:
        raise KeyError(
            f"[{label}] Could not find team columns for join. Tried home={_HOME_TEAM_CANDIDATES}, away={_AWAY_TEAM_CANDIDATES}. "
            f"Available: {list(df.columns)}"
        )

    if home != "home_team":
        df = df.rename(columns={home: "home_team"})
    if away != "away_team":
        df = df.rename(columns={away: "away_team"})

    # Normalize team strings a bit (trim)
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    df = _ensure_date_col(df, label=label)
    return df, ["date", "home_team", "away_team"]


# -----------------------------
# Core loading
# -----------------------------


def _load_predictions(pred_dir: str, pattern: str) -> pd.DataFrame:
    pred_dir_p = Path(pred_dir)
    paths = sorted(pred_dir_p.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files found in {pred_dir} with pattern {pattern}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["_pred_file"] = p.name
        frames.append(df)

    preds = pd.concat(frames, ignore_index=True)
    preds = _ensure_date_col(preds, label="preds")
    preds, join_keys = _ensure_join_keys(preds, label="preds")

    # Keep join keys as explicit columns even if file naming implied date.
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

    # actuals
    out["home_margin"] = out["home_score"] - out["away_score"]
    out["total_points"] = out["home_score"] + out["away_score"]
    out["home_win"] = (out["home_margin"] > 0).astype(int)

    # prob calibration / brier-ish
    if prob_col in out.columns:
        p = pd.to_numeric(out[prob_col], errors="coerce").clip(0, 1)
        out["prob"] = p
        out["logloss_component"] = -(
            out["home_win"] * np.log(np.clip(p, 1e-9, 1 - 1e-9))
            + (1 - out["home_win"]) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9))
        )
        out["brier_component"] = (p - out["home_win"]) ** 2

    # spread error
    if spread_col in out.columns:
        s = pd.to_numeric(out[spread_col], errors="coerce")
        out["pred_spread"] = s
        out["spread_error"] = (s - out["home_margin"]).abs()

    # total error
    if total_col in out.columns:
        t = pd.to_numeric(out[total_col], errors="coerce")
        out["pred_total"] = t
        out["total_error"] = (t - out["total_points"]).abs()

    return out


def _summarize(joined: pd.DataFrame) -> Dict[str, float]:
    def _mean(series_name: str) -> float:
        if series_name not in joined.columns:
            return float("nan")
        return float(pd.to_numeric(joined[series_name], errors="coerce").dropna().mean())

    metrics = {
        "n_rows": float(len(joined)),
        "n_complete_scores": float(joined[["home_score", "away_score"]].dropna().shape[0]),
        "mean_logloss": _mean("logloss_component"),
        "mean_brier": _mean("brier_component"),
        "mae_spread": _mean("spread_error"),
        "mae_total": _mean("total_error"),
    }
    return metrics


# -----------------------------
# CLI
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest model predictions vs historical results")
    p.add_argument("--pred-dir", required=True, help="Directory containing prediction CSVs")
    p.add_argument("--history", required=True, help="Historical games/results CSV")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--pattern", default="predictions_*.csv", help="Glob pattern for prediction files")
    p.add_argument("--prob-col", default="home_win_prob", help="Probability column name")
    p.add_argument("--spread-col", default="fair_spread", help="Spread prediction column name")
    p.add_argument("--total-col", default="fair_total", help="Total prediction column name")
    p.add_argument("--out-dir", default=None, help="Output dir (defaults to pred-dir)")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    pred_dir = args.pred_dir
    out_dir = args.out_dir or pred_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(pred_dir, args.pattern)
    hist = _load_history(args.history)

    # Filter date range
    preds = preds[(preds["date"] >= start) & (preds["date"] <= end)].copy()
    hist = hist[(hist["date"] >= start) & (hist["date"] <= end)].copy()

    # Join mode is determined by what keys exist in both.
    if "game_id" in preds.columns and "game_id" in hist.columns:
        join_keys = ["game_id"]
    else:
        join_keys = ["date", "home_team", "away_team"]

    joined = preds.merge(hist, how="inner", on=join_keys, suffixes=("", "_hist"))

    audit = {
        "pred_dir": pred_dir,
        "pattern": args.pattern,
        "history": args.history,
        "start": str(start),
        "end": str(end),
        "pred_rows": int(len(preds)),
        "hist_rows": int(len(hist)),
        "join_keys": join_keys,
        "joined_rows": int(len(joined)),
        "pred_cols": list(preds.columns),
        "hist_cols": list(hist.columns),
    }

    audit_path = Path(out_dir) / "backtest_join_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[backtest] wrote {audit_path}")

    if joined.empty:
        raise RuntimeError(
            "Backtest join produced 0 rows. Check join keys and team canonicalization. "
            f"See audit: {audit_path}"
        )

    joined_scored = _compute_metrics(joined, args.prob_col, args.spread_col, args.total_col)
    summary = _summarize(joined_scored)

    joined_path = Path(out_dir) / "backtest_joined.csv"
    joined_scored.to_csv(joined_path, index=False)
    print(f"[backtest] wrote {joined_path}")

    metrics_path = Path(out_dir) / "backtest_metrics.csv"
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    print(f"[backtest] wrote {metrics_path}")


if __name__ == "__main__":
    main()
