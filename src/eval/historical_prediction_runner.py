from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import _merge_key

# Directories for input and output
SNAPSHOT_DIR = Path("data/_snapshots")
OUTPUT_DIR = Path("outputs")

logger = logging.getLogger("historical")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [historical] %(message)s",
)

# Coverage thresholds for market ensemble enablement
COVERAGE_WARN_THRESHOLD = 0.95
COVERAGE_DISABLE_THRESHOLD = 0.80


# -------------------------
# deterministic JSON writer
# -------------------------


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


# -------------------------
# core helpers
# -------------------------


def load_history(history_path: Path) -> pd.DataFrame:
    df = pd.read_csv(history_path)
    if df.empty:
        raise ValueError(f"History CSV is empty: {history_path}")
    required = {"game_date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"History CSV missing required columns: {sorted(missing)}")
    return df


def load_games_for_date(history_df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Extract games for a specific run_date from the history DataFrame."""
    day = history_df[history_df["game_date"].astype(str) == run_date].copy()
    if day.empty:
        return pd.DataFrame(columns=["game_date", "home_team", "away_team"])

    # Only what predict_games needs
    out = day[["game_date", "home_team", "away_team"]].copy()
    out["game_date"] = run_date
    return out


def _find_latest_snapshot_for_date(run_date: str, snapshot_type: str = "close") -> Optional[Path]:
    """
    Find the latest odds snapshot file for a date.

    Expected naming convention:
      data/_snapshots/{snapshot_type}_{run_date}_*.csv
    """
    pattern = str(SNAPSHOT_DIR / f"{snapshot_type}_{run_date}_*.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    return Path(candidates[-1])


def _coverage_from_merge_keys(preds: pd.DataFrame, odds: pd.DataFrame) -> float:
    """
    Behavior-preserving coverage check between predicted games and odds snapshot.

    Coverage = (# of pred merge_keys found in odds merge_keys) / (# pred keys)
    """
    pred_keys = set(preds["merge_key"].astype(str).unique())
    odds_keys = set(odds["merge_key"].astype(str).unique())
    if not pred_keys:
        return 0.0
    return len(pred_keys & odds_keys) / len(pred_keys)


def run_day(
    history_df: pd.DataFrame,
    run_date: str,
    apply_market: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    """
    Run predictions for a single day.

    - Writes base predictions: outputs/predictions_{run_date}.csv
    - Optionally writes market preds: outputs/predictions_{run_date}_market.csv
      (only if odds snapshot exists AND coverage passes thresholds)

    Commit 2 addition:
    - returns a small audit dict (status, reason, coverage, paths written)
      without changing prediction semantics.
    """
    audit: Dict[str, Any] = {
        "date": run_date,
        "status": "unknown",
        "reason": None,
        "base_written": False,
        "market_written": False,
        "coverage": None,
        "snapshot_path": None,
    }

    games = load_games_for_date(history_df, run_date)
    if games.empty:
        logger.info("No games for %s — skipping", run_date)
        audit["status"] = "skipped"
        audit["reason"] = "no_games"
        return audit

    preds = predict_games(games)

    preds["merge_key"] = preds.apply(
        lambda r: _merge_key(r["home_team"], r["away_team"], r["game_date"]),
        axis=1,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    base_path = OUTPUT_DIR / f"predictions_{run_date}.csv"

    if base_path.exists() and not overwrite:
        logger.info("Base predictions already exist for %s — skipping write", run_date)
        audit["base_written"] = False
    else:
        preds.to_csv(base_path, index=False)
        logger.info("Wrote %s (%d rows)", base_path, len(preds))
        audit["base_written"] = True

    if not apply_market:
        audit["status"] = "ok"
        audit["reason"] = "base_only"
        return audit

    snap_path = _find_latest_snapshot_for_date(run_date, snapshot_type="close")
    if snap_path is None:
        logger.warning(
            "No CLOSE odds snapshot found for %s in %s — skipping market ensemble",
            run_date,
            SNAPSHOT_DIR,
        )
        audit["status"] = "skipped"
        audit["reason"] = "no_snapshot"
        return audit

    odds = pd.read_csv(snap_path)
    audit["snapshot_path"] = str(snap_path)

    if odds.empty or "merge_key" not in odds.columns:
        logger.warning(
            "CLOSE snapshot invalid for %s (%s): empty=%s, has_merge_key=%s — skipping market ensemble",
            run_date,
            snap_path,
            odds.empty,
            "merge_key" in odds.columns,
        )
        audit["status"] = "skipped"
        audit["reason"] = "invalid_snapshot"
        return audit

    coverage = _coverage_from_merge_keys(preds, odds)
    audit["coverage"] = float(coverage)

    if coverage < COVERAGE_DISABLE_THRESHOLD:
        logger.warning(
            "Coverage %.1f%% < %.0f%% for %s — disabling market ensemble for this day",
            coverage * 100.0,
            COVERAGE_DISABLE_THRESHOLD * 100.0,
            run_date,
        )
        audit["status"] = "skipped"
        audit["reason"] = "coverage_disabled"
        return audit
    elif coverage < COVERAGE_WARN_THRESHOLD:
        logger.warning(
            "Coverage %.1f%% < %.0f%% for %s — proceeding, but results may be incomplete",
            coverage * 100.0,
            COVERAGE_WARN_THRESHOLD * 100.0,
            run_date,
        )

    market_preds = apply_market_ensemble(preds, odds)

    out_path = OUTPUT_DIR / f"predictions_{run_date}_market.csv"
    if out_path.exists() and not overwrite:
        logger.info("Market predictions already exist for %s — skipping write", run_date)
        audit["status"] = "ok"
        audit["reason"] = "market_exists"
        audit["market_written"] = False
        return audit

    market_preds.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(market_preds))

    audit["status"] = "ok"
    audit["reason"] = "market_written"
    audit["market_written"] = True
    return audit


def _date_range(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    out = []
    d = d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main() -> None:
    """
    Usage:
      python -m src.eval.historical_prediction_runner \
        --history data/history/games_2019_2024.csv \
        --start 2023-10-24 \
        --end 2024-06-17 \
        --apply-market \
        --overwrite

    History CSV must contain:
      - game_date
      - home_team
      - away_team
    (It may contain scores too; we ignore them here.)
    """
    ap = argparse.ArgumentParser(
        description="Run historical model predictions (optionally market-adjusted) for a date range."
    )
    ap.add_argument("--history", required=True, help="Path to history games CSV (must include game_date, home_team, away_team).")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    ap.add_argument("--apply-market", action="store_true", help="Apply market ensemble using CLOSE odds snapshots if available.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    args = ap.parse_args()

    history_path = Path(args.history)
    history_df = load_history(history_path)

    dates = _date_range(args.start, args.end)

    # Commit 2: structured run audit
    run_audit: Dict[str, Any] = {
        "kind": "historical_prediction_runner_audit",
        "history_path": str(history_path),
        "start": args.start,
        "end": args.end,
        "apply_market": bool(args.apply_market),
        "overwrite": bool(args.overwrite),
        "snapshot_dir": str(SNAPSHOT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "coverage_warn_threshold": float(COVERAGE_WARN_THRESHOLD),
        "coverage_disable_threshold": float(COVERAGE_DISABLE_THRESHOLD),
        "days": [],
    }

    counts: Dict[str, int] = {}

    for d in dates:
        day_audit = run_day(history_df, d, apply_market=args.apply_market, overwrite=args.overwrite)
        run_audit["days"].append(day_audit)

        key = f'{day_audit.get("status","unknown")}::{day_audit.get("reason","unknown")}'
        counts[key] = counts.get(key, 0) + 1

    run_audit["summary"] = {
        "n_days": len(dates),
        "counts": counts,
        "n_ok": sum(v for k, v in counts.items() if k.startswith("ok::")),
        "n_skipped": sum(v for k, v in counts.items() if k.startswith("skipped::")),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    audit_path = OUTPUT_DIR / "historical_prediction_runner_audit.json"
    _write_json(audit_path, run_audit)
    logger.info("Wrote run audit to %s", audit_path)


if __name__ == "__main__":
    main()
