"""Backtest predictions against historical results.

Commit-2 goal:
- Join per-day prediction CSVs produced by historical_prediction_runner with a results/history CSV.
- Be resilient to different column names across data sources (balldontlie, custom exports).
- Never crash on missing score columns; emit an audit + metrics with clear warnings.

Expected prediction columns:
- game_id (preferred) or id
- home_team / away_team (optional)
- home_win_prob, fair_spread, fair_total (configurable via CLI)

Expected results/history columns:
- game_id or id
- date (or game_date)
- final scores (column names vary; we auto-detect)
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_DIR / "outputs"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    if "game_date" in df.columns:
        df["date"] = pd.to_datetime(df["game_date"]).dt.date
        return df
    if "start_time" in df.columns:
        df["date"] = pd.to_datetime(df["start_time"]).dt.date
        return df
    return df


def _pick_id_col(df: pd.DataFrame) -> str:
    if "game_id" in df.columns:
        return "game_id"
    if "id" in df.columns:
        return "id"
    raise ValueError(f"No join id column found. Columns={list(df.columns)}")


def _resolve_score_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Try to find home/away final score columns in a merged dataframe."""
    home_candidates = [
        "home_score", "home_team_score", "home_points", "home_pts",
        "home_score_hist", "home_team_score_hist", "home_points_hist", "home_pts_hist",
        "home_score_y", "home_team_score_y", "home_points_y", "home_pts_y",
    ]
    away_candidates = [
        "away_score", "visitor_score", "away_team_score", "away_points", "away_pts",
        "away_score_hist", "visitor_score_hist", "away_team_score_hist", "away_points_hist", "away_pts_hist",
        "away_score_y", "visitor_score_y", "away_team_score_y", "away_points_y", "away_pts_y",
    ]

    home = next((c for c in home_candidates if c in df.columns), None)
    away = next((c for c in away_candidates if c in df.columns), None)
    return home, away


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_date_col(df)

    # normalize id naming
    if "game_id" not in df.columns and "id" in df.columns:
        pass
    elif "id" not in df.columns and "game_id" in df.columns:
        pass

    return df


def _load_predictions(pred_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(Path(pred_dir).glob(pattern))
    if not files:
        # fallback patterns
        for pat in ["predictions_*.csv", "predictions_*_market.csv"]:
            files = sorted(Path(pred_dir).glob(pat))
            if files:
                break

    if not files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir} with pattern {pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__pred_file"] = str(f)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        raise FileNotFoundError(f"Could not read any prediction CSVs from {pred_dir}")

    out = pd.concat(dfs, ignore_index=True)
    return out


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="Directory containing prediction CSVs.")
    ap.add_argument("--pattern", default="predictions_*.csv", help="Glob pattern for prediction files.")
    ap.add_argument("--history", dest="history", default=None, help="Results/history CSV path (alias for --results).")
    ap.add_argument("--results", dest="results", default=None, help="Results CSV path (historical games).")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    ap.add_argument("--prob-col", default="home_win_prob", help="Probability column for win-prob metrics.")
    ap.add_argument("--spread-col", default="fair_spread", help="Spread column for error metrics.")
    ap.add_argument("--total-col", default="fair_total", help="Total column for error metrics.")
    ap.add_argument("--metrics-path", default=str(DEFAULT_OUT_DIR / "backtest_metrics.json"))
    ap.add_argument("--calib-path", default=str(DEFAULT_OUT_DIR / "backtest_calibration.csv"))
    ap.add_argument("--per-game-path", default=str(DEFAULT_OUT_DIR / "backtest_per_game.csv"))
    args = ap.parse_args()

    results_path = Path(args.history or args.results) if (args.history or args.results) else None
    if results_path is None:
        raise SystemExit("Must provide --history or --results")

    pred_dir = Path(args.pred_dir)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    preds = _load_predictions(pred_dir, args.pattern)
    results = _load_results(results_path)

    preds_id = _pick_id_col(preds)
    results_id = _pick_id_col(results)

    # Normalize ids to string to avoid int/str mismatches
    preds[preds_id] = preds[preds_id].astype(str)
    results[results_id] = results[results_id].astype(str)

    # Filter results to date range if possible
    if "date" in results.columns:
        results = results[(results["date"] >= start) & (results["date"] <= end)].copy()

    # Merge: keep history score columns clearly
    joined = preds.merge(
        results,
        how="left",
        left_on=preds_id,
        right_on=results_id,
        suffixes=("_pred", "_hist"),
    )

    # Basic join audit
    join_ok = joined[~joined[results_id].isna()].copy()
    audit = {
        "pred_dir": str(pred_dir),
        "pattern": args.pattern,
        "results": str(results_path),
        "start": args.start,
        "end": args.end,
        "pred_rows": int(len(preds)),
        "results_rows": int(len(results)),
        "joined_rows": int(len(joined)),
        "joined_ok_rows": int(len(join_ok)),
        "missing_results_rows": int(len(joined) - len(join_ok)),
        "prob_col": args.prob_col,
        "spread_col": args.spread_col,
        "total_col": args.total_col,
    }
    audit_path = Path(args.metrics_path).with_name("backtest_join_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[backtest] wrote {audit_path}")

    # Resolve final score cols
    home_sc, away_sc = _resolve_score_cols(join_ok)
    if home_sc is None or away_sc is None:
        metrics = {
            "status": "missing_score_columns",
            "message": "Could not auto-detect home/away score columns in joined data; metrics skipped.",
            "home_score_col": home_sc,
            "away_score_col": away_sc,
            "n_games": int(len(join_ok)),
        }
        Path(args.metrics_path).write_text(json.dumps(metrics, indent=2))
        print(f"[backtest][WARNING] Missing score columns; wrote {args.metrics_path} and exiting 0.")
        return

    # Build targets
    home_score = pd.to_numeric(join_ok[home_sc], errors="coerce")
    away_score = pd.to_numeric(join_ok[away_sc], errors="coerce")
    valid_scores = home_score.notna() & away_score.notna()
    join_ok = join_ok[valid_scores].copy()
    home_score = home_score[valid_scores]
    away_score = away_score[valid_scores]

    if len(join_ok) == 0:
        metrics = {
            "status": "no_scored_games",
            "message": "Joined rows exist but none have valid numeric final scores; metrics skipped.",
            "home_score_col": home_sc,
            "away_score_col": away_sc,
            "n_games": 0,
        }
        Path(args.metrics_path).write_text(json.dumps(metrics, indent=2))
        print(f"[backtest][WARNING] No scored games; wrote {args.metrics_path} and exiting 0.")
        return

    y_win = (home_score > away_score).astype(int).to_numpy()

    per_game = join_ok[[c for c in join_ok.columns if c not in []]].copy()
    per_game["home_score_final"] = home_score.to_numpy()
    per_game["away_score_final"] = away_score.to_numpy()
    per_game["home_win_actual"] = y_win

    metrics: Dict[str, object] = {
        "status": "ok",
        "n_games": int(len(join_ok)),
        "home_score_col": home_sc,
        "away_score_col": away_sc,
    }

    # Win prob metrics
    if args.prob_col in join_ok.columns:
        p = pd.to_numeric(join_ok[args.prob_col], errors="coerce").clip(0, 1)
        ok = p.notna()
        if ok.any():
            metrics["brier"] = _brier(y_win[ok.to_numpy()], p[ok].to_numpy())
            metrics["logloss"] = float(
                -np.mean(y_win[ok.to_numpy()] * np.log(np.clip(p[ok].to_numpy(), 1e-9, 1 - 1e-9)) +
                         (1 - y_win[ok.to_numpy()]) * np.log(np.clip(1 - p[ok].to_numpy(), 1e-9, 1 - 1e-9)))
            )
            per_game["p_home_win"] = p
        else:
            metrics["brier"] = None
            metrics["logloss"] = None
    else:
        metrics["brier"] = None
        metrics["logloss"] = None

    # Spread metrics (model fair spread vs actual margin)
    if args.spread_col in join_ok.columns:
        fair_spread = pd.to_numeric(join_ok[args.spread_col], errors="coerce")
        actual_margin = (home_score - away_score).to_numpy()
        ok = fair_spread.notna().to_numpy()
        if ok.any():
            metrics["spread_rmse"] = _rmse(fair_spread.to_numpy()[ok], actual_margin[ok])
            metrics["spread_mae"] = float(np.mean(np.abs(fair_spread.to_numpy()[ok] - actual_margin[ok])))
            per_game["actual_margin"] = actual_margin
            per_game["fair_spread"] = fair_spread
        else:
            metrics["spread_rmse"] = None
            metrics["spread_mae"] = None
    else:
        metrics["spread_rmse"] = None
        metrics["spread_mae"] = None

    # Total metrics (model fair total vs actual total)
    if args.total_col in join_ok.columns:
        fair_total = pd.to_numeric(join_ok[args.total_col], errors="coerce")
        actual_total = (home_score + away_score).to_numpy()
        ok = fair_total.notna().to_numpy()
        if ok.any():
            metrics["total_rmse"] = _rmse(fair_total.to_numpy()[ok], actual_total[ok])
            metrics["total_mae"] = float(np.mean(np.abs(fair_total.to_numpy()[ok] - actual_total[ok])))
            per_game["actual_total"] = actual_total
            per_game["fair_total"] = fair_total
        else:
            metrics["total_rmse"] = None
            metrics["total_mae"] = None
    else:
        metrics["total_rmse"] = None
        metrics["total_mae"] = None

    # Calibration table (10 bins)
    calib_path = Path(args.calib_path)
    if args.prob_col in join_ok.columns and "p_home_win" in per_game.columns:
        p = per_game["p_home_win"].astype(float)
        bins = np.linspace(0, 1, 11)
        per_game["prob_bin"] = pd.cut(p, bins=bins, include_lowest=True)
        calib = (
            per_game.groupby("prob_bin", observed=False)
            .agg(n=("home_win_actual", "size"), p_mean=("p_home_win", "mean"), win_rate=("home_win_actual", "mean"))
            .reset_index()
        )
        calib.to_csv(calib_path, index=False)
    else:
        # write empty file for consistent downstream expectations
        pd.DataFrame(columns=["prob_bin", "n", "p_mean", "win_rate"]).to_csv(calib_path, index=False)

    # Write per-game and metrics
    Path(args.per_game_path).parent.mkdir(parents=True, exist_ok=True)
    per_game.to_csv(Path(args.per_game_path), index=False)
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2))
    print(f"[backtest] wrote {args.metrics_path}")


if __name__ == "__main__":
    main()

