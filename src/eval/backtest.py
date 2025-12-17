"""
Backtesting engine for NBA Pro-Lite / Elite with merge-coverage hardening.

This module evaluates model and market-blended predictions against
historical results. It loads prediction CSV files (typically
``predictions_YYYY-MM-DD_market.csv``), merges them with game results on
canonical keys, computes standard metrics (Brier score, log loss,
calibration, spread MAE/RMSE) and writes out summary JSON/CSV files.

Key enhancements compared to earlier versions:

* **Merge coverage enforcement** – After merging predictions and results
  on normalized keys, the engine computes coverage (games with both
  predictions and results) versus the total number of games in the
  results for the specified date range. Coverage <80% raises a
  ``RuntimeError``; coverage between 80% and 95% triggers a warning.
  This prevents silent evaluation of incomplete data.

* **Coverage reporting** – The summary metrics now include
  ``n_games_total`` (total games in results), ``n_games_covered`` (games
  evaluated) and ``coverage`` (fraction). This contextualizes all
  performance metrics and aids reproducibility.

* **Transparent logging** – Each major step logs progress using the
  ``[backtest]`` logger prefix.

Commit 2 additions (behavior-preserving):
* **Join audit JSON** – Writes `outputs/backtest_join_audit.json` with
  merge coverage and sample missing keys on both sides of the join.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.team_names import normalize_team_name

logger = logging.getLogger("backtest")


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.
    """

    predictions_dir: str = "outputs"
    predictions_pattern: str = "predictions_*_market.csv"

    # Historical results file with final scores.
    # Expected columns:
    #   - game_date (YYYY-MM-DD)
    #   - home_team
    #   - away_team
    #   - home_score
    #   - away_score
    results_path: str = "data/history/games_2019_2024.csv"

    # Date range (inclusive). If None, uses all predictions found.
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Which probability and spread columns to evaluate
    prob_col: str = "home_win_prob"        # can switch to "home_win_prob_market"
    fair_spread_col: str = "fair_spread"   # can switch to "fair_spread_market"

    # Number of bins to use for calibration
    n_calibration_bins: int = 10

    # Output files
    metrics_path: str = os.path.join("outputs", "backtest_metrics.json")
    calibration_path: str = os.path.join("outputs", "backtest_calibration.csv")
    per_game_path: str = os.path.join("outputs", "backtest_per_game.csv")


@dataclass
class BacktestMetrics:
    """
    Summary metrics for the backtest run.
    """

    brier_score: float
    log_loss: float
    mean_pred_prob: float
    base_rate_home_win: float
    spread_mae: float
    spread_rmse: float
    avg_margin: float

    n_games_total: int
    n_games_covered: int
    coverage: float


# ---------------------------------------------------------------------
# UTIL: deterministic JSON writer
# ---------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------
# DATA LOADING / KEYING
# ---------------------------------------------------------------------


def _add_normalized_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team_norm"] = df["home_team"].astype(str).map(normalize_team_name)
    df["away_team_norm"] = df["away_team"].astype(str).map(normalize_team_name)
    df["game_date_norm"] = df["game_date"].astype(str)
    df["merge_key_norm"] = (
        df["home_team_norm"].astype(str)
        + "__"
        + df["away_team_norm"].astype(str)
        + "__"
        + df["game_date_norm"].astype(str)
    )
    return df


def load_predictions_history(cfg: BacktestConfig) -> pd.DataFrame:
    pred_glob = os.path.join(cfg.predictions_dir, cfg.predictions_pattern)
    files = sorted(glob.glob(pred_glob))
    if not files:
        raise FileNotFoundError(f"No prediction files found matching {pred_glob}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        if df.empty:
            continue
        rows.append(df)

    if not rows:
        raise ValueError("All prediction files were empty.")
    preds = pd.concat(rows, ignore_index=True)

    # Optional date range filter
    if cfg.start_date:
        preds = preds[preds["game_date"].astype(str) >= cfg.start_date]
    if cfg.end_date:
        preds = preds[preds["game_date"].astype(str) <= cfg.end_date]

    logger.info("[backtest] Loaded predictions: %d rows from %d files.", len(preds), len(files))
    return preds


def load_results(cfg: BacktestConfig) -> pd.DataFrame:
    results = pd.read_csv(cfg.results_path)
    if results.empty:
        raise ValueError(f"Results file is empty: {cfg.results_path}")

    # Optional date range filter
    if cfg.start_date:
        results = results[results["game_date"].astype(str) >= cfg.start_date]
    if cfg.end_date:
        results = results[results["game_date"].astype(str) <= cfg.end_date]

    logger.info("[backtest] Loaded results: %d rows from %s.", len(results), cfg.results_path)
    return results


def merge_predictions_and_results(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    preds_keyed = _add_normalized_keys(preds)
    results_keyed = _add_normalized_keys(results)

    merged = preds_keyed.merge(
        results_keyed[
            [
                "merge_key_norm",
                "home_score",
                "away_score",
            ]
        ],
        on="merge_key_norm",
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "No overlap between predictions and results after merging on normalized keys. "
            "Check that team naming and dates are consistent."
        )
    logger.info(
        "[backtest] Merged to %d rows (games with both predictions and results).",
        len(merged),
    )
    return merged


def compute_merge_coverage(
    preds: pd.DataFrame, results: pd.DataFrame
) -> Tuple[int, int, float]:
    """
    Compute merge coverage statistics.

    Returns a tuple: (n_games_covered, n_games_total, coverage_ratio)
    where:
      - n_games_covered: number of unique merge keys present in both predictions and results
      - n_games_total:   number of unique merge keys in the results within the date range
      - coverage_ratio:  n_games_covered / n_games_total

    Team names and game dates are normalized via ``normalize_team_name``.
    """
    preds_keyed = _add_normalized_keys(preds)
    results_keyed = _add_normalized_keys(results)
    pred_keys = set(preds_keyed["merge_key_norm"].unique())
    result_keys = set(results_keyed["merge_key_norm"].unique())
    n_games_total = len(result_keys)
    n_games_covered = len(pred_keys & result_keys)
    coverage_ratio = n_games_covered / n_games_total if n_games_total > 0 else 0.0
    return n_games_covered, n_games_total, coverage_ratio


def _join_audit(preds: pd.DataFrame, results: pd.DataFrame, sample_n: int = 10) -> dict:
    """
    Behavior-preserving: computes diagnostics only.
    """
    preds_keyed = _add_normalized_keys(preds)
    results_keyed = _add_normalized_keys(results)

    pred_keys = set(preds_keyed["merge_key_norm"].unique())
    result_keys = set(results_keyed["merge_key_norm"].unique())

    missing_in_preds = sorted(list(result_keys - pred_keys))[:sample_n]
    missing_in_results = sorted(list(pred_keys - result_keys))[:sample_n]

    n_cov, n_total, cov = compute_merge_coverage(preds, results)

    return {
        "n_games_total": int(n_total),
        "n_games_covered": int(n_cov),
        "coverage": float(cov),
        "missing_in_predictions_sample": missing_in_preds,
        "missing_in_results_sample": missing_in_results,
    }


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------


def _calibration_bins(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int
) -> pd.DataFrame:
    df = pd.DataFrame({"prob": probs, "y": y_true})
    df["bin"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop")
    out = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "count"), mean_prob=("prob", "mean"), emp_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    return out


def compute_metrics(
    df: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[float, float, float, float, float, float, float, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Actual home win indicator
    df["home_win_actual"] = (df["home_score"] > df["away_score"]).astype(int)

    # Predicted probability (clipped for log-loss stability)
    prob = df[cfg.prob_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    y = df["home_win_actual"].to_numpy()

    brier = float(brier_score_loss(y, prob))
    ll = float(log_loss(y, prob))
    mean_prob = float(prob.mean())
    base_rate = float(y.mean())

    # Spread error if columns exist
    if cfg.fair_spread_col in df.columns:
        fair_spread = df[cfg.fair_spread_col].astype(float).to_numpy()
        actual_margin = (df["home_score"] - df["away_score"]).astype(float).to_numpy()
        spread_err = fair_spread - actual_margin
        spread_mae = float(np.mean(np.abs(spread_err)))
        spread_rmse = float(np.sqrt(np.mean(spread_err**2)))
        avg_margin = float(np.mean(actual_margin))
    else:
        spread_mae = float("nan")
        spread_rmse = float("nan")
        avg_margin = float("nan")

    calib_df = _calibration_bins(prob, y, cfg.n_calibration_bins)

    # Per-game table for downstream QA and modeling
    per_game_df = df[
        [
            "game_date_norm",
            "home_team_norm",
            "away_team_norm",
            "merge_key_norm",
            cfg.prob_col,
            "home_score",
            "away_score",
            "home_win_actual",
        ]
    ].copy()
    per_game_df["actual_margin"] = (df["home_score"] - df["away_score"]).astype(float)

    if cfg.fair_spread_col in df.columns:
        per_game_df[cfg.fair_spread_col] = df[cfg.fair_spread_col].astype(float)
        per_game_df["spread_error"] = (
            per_game_df[cfg.fair_spread_col] - per_game_df["actual_margin"]
        )

    return (
        brier,
        ll,
        mean_prob,
        base_rate,
        spread_mae,
        spread_rmse,
        avg_margin,
        calib_df,
        per_game_df,
    )


# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------


def run_backtest(cfg: BacktestConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [backtest] %(message)s",
    )
    logger.info("Starting backtest with config: %s", cfg)

    preds = load_predictions_history(cfg)
    results = load_results(cfg)

    # Join audit (behavior-preserving): write before enforcing thresholds
    audit = _join_audit(preds, results, sample_n=10)
    out_dir = Path(cfg.metrics_path).resolve().parent
    audit_path = out_dir / "backtest_join_audit.json"
    audit_payload = {
        "kind": "backtest_join_audit",
        "predictions_dir": cfg.predictions_dir,
        "predictions_pattern": cfg.predictions_pattern,
        "results_path": cfg.results_path,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        **audit,
    }
    _write_json(audit_path, audit_payload)
    logger.info("Wrote join audit to %s", audit_path)

    merged = merge_predictions_and_results(preds, results)

    # Compute merge coverage and enforce thresholds
    n_cov, n_total, coverage_ratio = (
        audit["n_games_covered"],
        audit["n_games_total"],
        audit["coverage"],
    )

    logger.info(
        "[backtest] Coverage: %d/%d games (%.1f%%).",
        n_cov,
        n_total,
        coverage_ratio * 100.0,
    )

    if coverage_ratio < 0.80:
        raise RuntimeError(
            f"Merge coverage too low: {coverage_ratio:.3f} ({n_cov}/{n_total}). "
            "Check merge key contract, team normalization, and prediction file completeness. "
            f"See {audit_path} for missing key samples."
        )
    elif coverage_ratio < 0.95:
        logger.warning(
            "[backtest] Merge coverage %.1f%% below warning threshold (95%%). "
            "Proceeding, but metrics may reflect incomplete coverage. "
            "See join audit JSON for missing key samples.",
            coverage_ratio * 100.0,
        )

    (
        brier,
        ll,
        mean_prob,
        base_rate,
        spread_mae,
        spread_rmse,
        avg_margin,
        calib_df,
        per_game_df,
    ) = compute_metrics(merged, cfg)

    metrics = BacktestMetrics(
        brier_score=brier,
        log_loss=ll,
        mean_pred_prob=mean_prob,
        base_rate_home_win=base_rate,
        spread_mae=spread_mae,
        spread_rmse=spread_rmse,
        avg_margin=avg_margin,
        n_games_total=int(n_total),
        n_games_covered=int(n_cov),
        coverage=float(coverage_ratio),
    )

    # Write summary metrics JSON
    Path(cfg.metrics_path).parent and Path(cfg.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info("Wrote backtest metrics to %s", cfg.metrics_path)

    # Write calibration CSV
    calib_df.to_csv(cfg.calibration_path, index=False)
    logger.info("Wrote calibration bins to %s", cfg.calibration_path)

    # Write per-game CSV
    per_game_df.to_csv(cfg.per_game_path, index=False)
    logger.info("Wrote per-game backtest data to %s", cfg.per_game_path)

    logger.info(
        "Backtest complete: brier=%.4f logloss=%.4f spread_mae=%.3f coverage=%.1f%%",
        brier,
        ll,
        spread_mae,
        coverage_ratio * 100.0,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args() -> BacktestConfig:
    ap = argparse.ArgumentParser(description="Run a backtest over historical predictions.")
    ap.add_argument("--pred-dir", default="outputs", help="Directory with prediction CSVs.")
    ap.add_argument(
        "--pattern",
        default="predictions_*_market.csv",
        help="Glob pattern for prediction files (inside --pred-dir).",
    )
    ap.add_argument(
        "--results",
        default="data/history/games_2019_2024.csv",
        help="Historical results CSV with scores.",
    )
    ap.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    ap.add_argument("--prob-col", default="home_win_prob", help="Probability column to score.")
    ap.add_argument("--spread-col", default="fair_spread", help="Spread column to score.")
    ap.add_argument("--bins", type=int, default=10, help="Number of calibration bins.")
    ap.add_argument(
        "--metrics-path",
        default=os.path.join("outputs", "backtest_metrics.json"),
        help="Where to write summary metrics JSON.",
    )
    ap.add_argument(
        "--calib-path",
        default=os.path.join("outputs", "backtest_calibration.csv"),
        help="Where to write calibration bins CSV.",
    )
    ap.add_argument(
        "--per-game-path",
        default=os.path.join("outputs", "backtest_per_game.csv"),
        help="Where to write per-game CSV.",
    )

    args = ap.parse_args()

    cfg = BacktestConfig(
        predictions_dir=args.pred_dir,
        predictions_pattern=args.pattern,
        results_path=args.results,
        start_date=args.start,
        end_date=args.end,
        prob_col=args.prob_col,
        fair_spread_col=args.spread_col,
        n_calibration_bins=args.bins,
        metrics_path=args.metrics_path,
        calibration_path=args.calib_path,
        per_game_path=args.per_game_path,
    )
    return cfg


def main() -> None:
    cfg = _parse_args()
    run_backtest(cfg)


if __name__ == "__main__":
    main()
