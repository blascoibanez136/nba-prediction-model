#!/usr/bin/env python3
"""
Backtest runner.

Commit 2 goal: join per-day prediction CSVs with historical outcomes, then compute a few
basic diagnostics/metrics without assuming fragile column names.

This script does NOT assume predictions contain actual scores. Scores come from history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_date_col(df: pd.DataFrame) -> str:
    c = _find_first(df, ["date", "game_date", "gameDate", "start_date"])
    if not c:
        raise ValueError(f"Could not find date column in history. Columns: {list(df.columns)[:60]}...")
    return c


def _normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out = out.dropna(subset=[date_col])
    return out


def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _parse_date_col(df)
    df = _normalize_dates(df, date_col).rename(columns={date_col: "date"})

    # teams
    ren = {}
    if "home_team" not in df.columns:
        hc = _find_first(df, ["homeTeam", "home", "team_home", "home_abbr"])
        if hc:
            ren[hc] = "home_team"
    if "away_team" not in df.columns:
        ac = _find_first(df, ["awayTeam", "away", "team_away", "visitor_team", "visitor_abbr"])
        if ac:
            ren[ac] = "away_team"

    # scores (many possible schemas)
    if "home_score" not in df.columns:
        hs = _find_first(df, ["home_score", "home_points", "home_pts", "pts_home", "home_team_score", "homeScore"])
        if hs:
            ren[hs] = "home_score"
    if "away_score" not in df.columns:
        as_ = _find_first(df, ["away_score", "away_points", "away_pts", "pts_away", "visitor_team_score", "awayScore"])
        if as_:
            ren[as_] = "away_score"

    df = df.rename(columns=ren)

    missing = [c for c in ["date", "home_team", "away_team", "home_score", "away_score"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"History missing required columns after standardization: {missing}. "
            f"Columns: {list(df.columns)[:80]}..."
        )

    # numeric scores
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    return df


def _load_predictions(pred_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(pred_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir} with pattern {pattern}")

    dfs = []
    for p in files:
        df = pd.read_csv(p)
        # normalize date + required cols
        date_col = _find_first(df, ["date", "game_date"])
        if not date_col:
            # If missing, infer from filename predictions_YYYY-MM-DD.csv
            m = p.stem.replace("predictions_", "")
            try:
                df["date"] = pd.to_datetime(m).date()
            except Exception:
                raise ValueError(f"{p}: missing date column and cannot infer from filename")
        else:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date

        # ensure team cols
        if "home_team" not in df.columns or "away_team" not in df.columns:
            raise ValueError(f"{p}: predictions must include home_team and away_team. Columns: {list(df.columns)[:80]}")

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["date", "home_team", "away_team"])
    return out


def _compute_metrics(joined: pd.DataFrame, prob_col: str, spread_col: str, total_col: str) -> dict:
    df = joined.copy()

    # Outcomes
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]

    metrics = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min()) if len(df) else None,
        "date_max": str(df["date"].max()) if len(df) else None,
    }

    # Win prob (Brier)
    if prob_col in df.columns:
        p = pd.to_numeric(df[prob_col], errors="coerce")
        metrics["brier_home_win"] = float(((p - df["home_win"]) ** 2).mean()) if p.notna().any() else None
        metrics["prob_nunique"] = int(p.nunique(dropna=True)) if p.notna().any() else 0
    else:
        metrics["brier_home_win"] = None
        metrics["prob_nunique"] = None

    # Spread MAE vs actual margin
    if spread_col in df.columns:
        s = pd.to_numeric(df[spread_col], errors="coerce")
        metrics["spread_mae"] = float((s - df["margin"]).abs().mean()) if s.notna().any() else None
        metrics["spread_nunique"] = int(s.nunique(dropna=True)) if s.notna().any() else 0
    else:
        metrics["spread_mae"] = None
        metrics["spread_nunique"] = None

    # Total MAE vs actual total
    if total_col in df.columns:
        t = pd.to_numeric(df[total_col], errors="coerce")
        metrics["total_mae"] = float((t - df["total_points"]).abs().mean()) if t.notna().any() else None
        metrics["total_nunique"] = int(t.nunique(dropna=True)) if t.notna().any() else 0
    else:
        metrics["total_mae"] = None
        metrics["total_nunique"] = None

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="Directory containing predictions_YYYY-MM-DD.csv files")
    ap.add_argument("--pattern", default="predictions_*.csv", help="Glob pattern for prediction files")
    ap.add_argument("--history", required=True, help="Historical games CSV")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--prob-col", default="home_win_prob")
    ap.add_argument("--spread-col", default="fair_spread")
    ap.add_argument("--total-col", default="fair_total")
    ap.add_argument("--out-csv", default="outputs/backtest_joined.csv")
    ap.add_argument("--audit-path", default="outputs/backtest_join_audit.json")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)
    audit_path = Path(args.audit_path)
    _ensure_dir(out_csv.parent)
    _ensure_dir(audit_path.parent)

    preds = _load_predictions(pred_dir, args.pattern)

    hist = pd.read_csv(args.history)
    hist = _standardize_history(hist)

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    preds = preds[(preds["date"] >= start) & (preds["date"] <= end)].copy()
    hist = hist[(hist["date"] >= start) & (hist["date"] <= end)].copy()

    joined = preds.merge(
        hist[["date", "home_team", "away_team", "home_score", "away_score"]],
        how="left",
        on=["date", "home_team", "away_team"],
        validate="m:1",
    )

    # Some histories might use swapped home/away naming; attempt a fallback join if too many missing.
    miss = joined["home_score"].isna().mean()
    if miss > 0.20:
        swapped = preds.merge(
            hist[["date", "home_team", "away_team", "home_score", "away_score"]].rename(
                columns={"home_team": "away_team", "away_team": "home_team", "home_score": "away_score", "away_score": "home_score"}
            ),
            how="left",
            on=["date", "home_team", "away_team"],
            validate="m:1",
        )
        if swapped["home_score"].isna().mean() < miss:
            joined = swapped

    joined.to_csv(out_csv, index=False)

    metrics = _compute_metrics(joined.dropna(subset=["home_score", "away_score"]), args.prob_col, args.spread_col, args.total_col)

    audit = {
        "pred_dir": str(pred_dir),
        "history": str(args.history),
        "start": str(start),
        "end": str(end),
        "pattern": args.pattern,
        "out_csv": str(out_csv),
        "metrics": metrics,
        "join_missing_rate": float(joined["home_score"].isna().mean()) if len(joined) else None,
        "columns_predictions": list(preds.columns),
        "columns_history_sample": list(hist.columns)[:120],
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[backtest] wrote {audit_path}")
    print(f"[backtest] wrote {out_csv}")


if __name__ == "__main__":
    main()

