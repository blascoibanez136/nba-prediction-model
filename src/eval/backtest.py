"""
Backtesting engine for NBA Pro-Lite / Elite with merge‑coverage hardening.

This module evaluates model and market‑blended predictions against
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
  ``[backtest]`` logger prefix. Coverage warnings/errors are clearly
  indicated.

Outputs:
    - ``outputs/backtest_metrics.json``   – summary metrics (JSON)
    - ``outputs/backtest_calibration.csv`` – calibration bins (CSV)
    - ``outputs/backtest_per_game.csv``    – per‑game errors (CSV)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ingest.team_normalizer import normalize_team_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CONFIG / DATA CLASSES
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
    Summary metrics for the backtest.
    """

    # Games evaluated
    n_games: int
    # Total games in results for the specified date range
    n_games_total: int
    # Coverage ratio (n_games / n_games_total)
    coverage: float
    # Which columns were evaluated
    prob_col: str
    fair_spread_col: str

    # Classification / probability metrics
    brier_score: float
    log_loss: float
    mean_pred_prob: float
    base_rate_home_win: float

    # Spread error (using fair_spread -> expected margin)
    spread_mae: float
    spread_rmse: float

    # Raw score distributions (optional for inspection)
    avg_home_score: float
    avg_away_score: float
    avg_margin: float


# ---------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------


def _parse_date_from_predictions_filename(path: str) -> Optional[str]:
    """
    Given a path like:
        outputs/predictions_2025-12-11_market.csv
    return:
        "2025-12-11"
    """
    base = os.path.basename(path)
    if not base.startswith("predictions_"):
        return None
    without_prefix = base.replace("predictions_", "")
    date_part = without_prefix.split("_market")[0]
    return date_part


def load_predictions_history(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Load and concatenate all predictions_*_market.csv files in the given
    directory and date range.

    Returns a DataFrame with at least:
        - game_date
        - home_team
        - away_team
        - cfg.prob_col
        - cfg.fair_spread_col
    """
    pattern = os.path.join(cfg.predictions_dir, cfg.predictions_pattern)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction files found matching {pattern}")
    logger.info("[backtest] Found %d prediction files.", len(files))
    dfs: List[pd.DataFrame] = []
    for path in sorted(files):
        date_str = _parse_date_from_predictions_filename(path)
        if date_str is None:
            continue
        if cfg.start_date and date_str < cfg.start_date:
            continue
        if cfg.end_date and date_str > cfg.end_date:
            continue
        df = pd.read_csv(path)
        # Ensure required columns exist
        df["game_date"] = df["game_date"].astype(str)
        df["pred_file_date"] = date_str
        dfs.append(df)
    if not dfs:
        raise ValueError(
            f"No prediction files within date range {cfg.start_date} .. {cfg.end_date}"
        )
    all_preds = pd.concat(dfs, ignore_index=True)
    required_cols = {
        "game_date",
        "home_team",
        "away_team",
        cfg.prob_col,
        cfg.fair_spread_col,
    }
    missing = required_cols - set(all_preds.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    logger.info(
        "[backtest] Loaded %d rows of predictions from %d files.",
        len(all_preds),
        len(dfs),
    )
    return all_preds


def load_results(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Load historical results CSV with final scores.

    Expected columns:
        - game_date
        - home_team
        - away_team
        - home_score
        - away_score
    """
    if not os.path.exists(cfg.results_path):
        raise FileNotFoundError(
            f"Results file not found at {cfg.results_path}. "
            "Make sure you have a CSV with columns: "
            "game_date, home_team, away_team, home_score, away_score."
        )
    results = pd.read_csv(cfg.results_path)
    required_cols = {
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing = required_cols - set(results.columns)
    if missing:
        raise ValueError(f"Results CSV missing required columns: {missing}")
    results["game_date"] = results["game_date"].astype(str)
    logger.info(
        "[backtest] Loaded %d rows of results from %s.", len(results), cfg.results_path
    )
    return results


# ---------------------------------------------------------------------
# MERGE PREDICTIONS + RESULTS
# ---------------------------------------------------------------------


def _add_normalized_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized team names and a canonical merge key to a DataFrame
    with columns:
        - game_date
        - home_team
        - away_team
    """
    out = df.copy()
    out["home_team_norm"] = out["home_team"].apply(normalize_team_name)
    out["away_team_norm"] = out["away_team"].apply(normalize_team_name)
    out["merge_key_norm"] = (
        out["home_team_norm"].astype(str)
        + "__"
        + out["away_team_norm"].astype(str)
        + "__"
        + out["game_date"].astype(str)
    )
    return out


def merge_predictions_and_results(
    preds: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge predictions with final results on normalized merge key.
    Returns only rows with both predictions and results.
    """
    preds_keyed = _add_normalized_keys(preds)
    results_keyed = _add_normalized_keys(results)
    merged = preds_keyed.merge(
        results_keyed[["merge_key_norm", "home_score", "away_score"]],
        on="merge_key_norm",
        how="inner",
        suffixes=("", "_res"),
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


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------


def compute_metrics(
    df: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[float, float, float, float, float, float, float, pd.DataFrame, pd.DataFrame]:
    """
    Compute backtest metrics given a DataFrame that already contains:

        - home_score
        - away_score
        - cfg.prob_col (e.g. home_win_prob or home_win_prob_market)
        - cfg.fair_spread_col (e.g. fair_spread or fair_spread_market)

    Returns:
        - brier_score
        - log_loss
        - mean_pred_prob
        - base_rate_home_win
        - spread_mae
        - spread_rmse
        - avg_margin
        - calib_df: calibration DataFrame
        - per_game_df: DataFrame with per‑game errors
    """
    df = df.copy()
    # Actual home win indicator
    df["home_win_actual"] = (df["home_score"] > df["away_score"]).astype(int)
    # Predicted probability (clipped for log-loss stability)
    p = df[cfg.prob_col].astype(float).clip(1e-6, 1 - 1e-6)
    y = df["home_win_actual"].astype(float)
    # Brier score
    brier = float(((p - y) ** 2).mean())
    # Log loss
    log_loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    # Base rates
    mean_p = float(p.mean())
    base_rate = float(y.mean())
    # Spread error: convert fair_spread to expected margin (home perspective)
    fair_spread = df[cfg.fair_spread_col].astype(float)
    expected_margin = -fair_spread
    actual_margin = df["home_score"] - df["away_score"]
    spread_error = expected_margin - actual_margin
    spread_mae = float(spread_error.abs().mean())
    spread_rmse = float(math.sqrt((spread_error ** 2).mean()))
    # Calibration bins
    n_bins = cfg.n_calibration_bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bin_edges, right=True)  # 1..n_bins
    df["prob_bin"] = bin_indices
    calib_rows = []
    for b in range(1, n_bins + 1):
        mask = df["prob_bin"] == b
        if not mask.any():
            continue
        bin_p = p[mask].mean()
        bin_y = y[mask].mean()
        count = int(mask.sum())
        calib_rows.append(
            {
                "bin": b,
                "bin_lower": bin_edges[b - 1],
                "bin_upper": bin_edges[b],
                "mean_pred_prob": float(bin_p),
                "empirical_win_rate": float(bin_y),
                "n_games": count,
            }
        )
    calib_df = pd.DataFrame(calib_rows)
    # Attach some useful per-game columns for optional investigation
    per_game_df = df.copy()
    per_game_df["expected_margin"] = expected_margin
    per_game_df["spread_error"] = spread_error
    per_game_df["actual_margin"] = actual_margin
    return (
        brier,
        log_loss,
        mean_p,
        base_rate,
        spread_mae,
        spread_rmse,
        float(actual_margin.mean()),
        calib_df,
        per_game_df,
    )


# ---------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------


def run_backtest(cfg: BacktestConfig) -> BacktestMetrics:
    """
    Run the full backtest and write outputs to disk:

        - cfg.metrics_path (JSON summary)
        - cfg.calibration_path (CSV with calibration bins)
        - cfg.per_game_path (CSV with per-game errors)

    Enforces merge coverage and logs warnings/errors accordingly.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [backtest] %(message)s",
    )
    logger.info("Starting backtest with config: %s", cfg)
    preds = load_predictions_history(cfg)
    results = load_results(cfg)
    merged = merge_predictions_and_results(preds, results)
    # Compute merge coverage and enforce thresholds
    n_cov, n_total, coverage_ratio = compute_merge_coverage(preds, results)
    logger.info(
        "[backtest] Coverage: %d/%d games (%.1f%%).", n_cov, n_total, coverage_ratio * 100.0
    )
    if coverage_ratio < 0.80:
        # Too few games covered; do not proceed
        raise RuntimeError(
            f"Backtest merge coverage {coverage_ratio:.1%} < 80%. "
            "Check that predictions and results share consistent team names and dates."
        )
    elif coverage_ratio < 0.95:
        logger.warning(
            "[backtest] Coverage %.1f%% < 95%%. Results may be biased. Proceeding anyway.",
            coverage_ratio * 100.0,
        )
    # Compute metrics on merged DataFrame
    (
        brier,
        log_loss,
        mean_p,
        base_rate,
        spread_mae,
        spread_rmse,
        avg_margin,
        calib_df,
        per_game_df,
    ) = compute_metrics(merged, cfg)
    metrics = BacktestMetrics(
        n_games=int(len(merged)),
        n_games_total=int(n_total),
        coverage=float(coverage_ratio),
        prob_col=cfg.prob_col,
        fair_spread_col=cfg.fair_spread_col,
        brier_score=brier,
        log_loss=log_loss,
        mean_pred_prob=mean_p,
        base_rate_home_win=base_rate,
        spread_mae=spread_mae,
        spread_rmse=spread_rmse,
        avg_home_score=float(per_game_df["home_score"].mean()),
        avg_away_score=float(per_game_df["away_score"].mean()),
        avg_margin=avg_margin,
    )
    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.metrics_path), exist_ok=True)
    # Write metrics JSON
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
        "Backtest complete: n_games=%d (total=%d, coverage=%.1f%%), Brier=%.4f, log_loss=%.4f, spread_MAE=%.3f",
        metrics.n_games,
        metrics.n_games_total,
        metrics.coverage * 100.0,
        metrics.brier_score,
        metrics.log_loss,
        metrics.spread_mae,
    )
    return metrics


def _parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Run NBA model backtest over historical games."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="data/history/games_2019_2024.csv",
        help="Path to CSV with historical results (with scores).",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="outputs",
        help="Directory with predictions_*_market.csv files.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) for backtest (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) for backtest (inclusive).",
    )
    parser.add_argument(
        "--prob-col",
        type=str,
        default="home_win_prob",
        help="Probability column to evaluate (e.g. 'home_win_prob_market').",
    )
    parser.add_argument(
        "--spread-col",
        type=str,
        default="fair_spread",
        help="Spread column to evaluate (e.g. 'fair_spread_market').",
    )
    args = parser.parse_args()
    cfg = BacktestConfig(
        predictions_dir=args.pred_dir,
        results_path=args.results,
        start_date=args.start,
        end_date=args.end,
        prob_col=args.prob_col,
        fair_spread_col=args.spread_col,
    )
    return cfg


def main() -> None:
    cfg = _parse_args()
    run_backtest(cfg)


if __name__ == "__main__":
    main()
