"""
Historical Picks Runner (Commit-4 / Option A)
--------------------------------------------

Purpose:
- Generate pick files across a historical date range from existing daily prediction CSVs.
- This is additive and does NOT change prediction/backtest core behavior.
- Mirrors src/eval/historical_prediction_runner.py but for picks.

Inputs:
- outputs/predictions_YYYY-MM-DD.csv (already produced by historical_prediction_runner)
- Uses existing pick logic module (Commit-3 conservative picks), with a safe fallback.

Outputs:
- outputs/picks_YYYY-MM-DD.csv
- outputs/audits/picks_YYYY-MM-DD_audit.json
- outputs/historical_picks_runner_audit.json (coverage summary)

Rules:
- Deterministic
- No silent drops (all skips logged)
- Does not modify predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------
# Config / Contracts
# -----------------------------

PRED_RE = re.compile(r"^predictions_(\d{4}-\d{2}-\d{2})\.csv$")


@dataclass(frozen=True)
class PicksRunnerConfig:
    prob_floor: float = 0.62
    max_picks_per_day: int = 3
    min_games_for_picks: int = 2
    # If you have calibration info in preds, these are safe defaults:
    require_calibration_keep: bool = False
    max_abs_gap: float = 0.08
    # Column names
    date_col: str = "game_date"
    prob_col: str = "home_win_prob"


# -----------------------------
# Import existing pick logic (preferred)
# -----------------------------

def _try_import_commit3_picks():
    """
    Try to import your existing conservative picks generator.
    We keep this flexible since file/module names vary between setups.
    """
    candidates = [
        ("src.picks.conservative_picks", "generate_conservative_picks"),
        ("src.picks.picks", "generate_conservative_picks"),
        ("src.picks.conservative", "generate_conservative_picks"),
    ]
    for mod, fn in candidates:
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)
        except Exception:
            continue
    return None


# -----------------------------
# Minimal fallback conservative picks (only used if import fails)
# -----------------------------

def _fallback_generate_conservative_picks(
    preds_df: pd.DataFrame,
    *,
    config: PicksRunnerConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fallback conservative picks:
    - ML HOME only
    - prob_floor
    - max_picks_per_day cap
    - deterministic ordering by prob desc
    """
    df = preds_df.copy()

    # normalize date column
    if config.date_col not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": config.date_col})
    if config.date_col not in df.columns:
        df[config.date_col] = ""

    required = {"home_team", "away_team", config.prob_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[historical_picks] Missing required columns in preds: {sorted(missing)}")

    n_games = int(len(df))
    if n_games < config.min_games_for_picks:
        audit = {
            "n_games": n_games,
            "n_candidates": 0,
            "n_picks": 0,
            "policy": {
                "prob_floor": config.prob_floor,
                "max_picks_per_day": config.max_picks_per_day,
                "min_games_for_picks": config.min_games_for_picks,
            },
            "reason": "min_games_for_picks not met",
        }
        return df.head(0), audit

    df["home_win_prob"] = pd.to_numeric(df[config.prob_col], errors="coerce").clip(0.0, 1.0)
    candidates = df[df["home_win_prob"] >= float(config.prob_floor)].copy()
    candidates = candidates.sort_values(["home_win_prob", "home_team", "away_team"], ascending=[False, True, True])

    picks = candidates.head(int(config.max_picks_per_day)).copy()
    if picks.empty:
        audit = {
            "n_games": n_games,
            "n_candidates": int(len(candidates)),
            "n_picks": 0,
            "policy": {
                "prob_floor": config.prob_floor,
                "max_picks_per_day": config.max_picks_per_day,
                "min_games_for_picks": config.min_games_for_picks,
            },
            "reason": "no candidates met prob_floor",
        }
        return picks, audit

    picks["pick_type"] = "ML"
    picks["pick_side"] = "HOME"
    picks["confidence"] = picks["home_win_prob"]
    picks["reason"] = picks["home_win_prob"].apply(
        lambda p: f"prob>={config.prob_floor:.2f};prob={float(p):.3f}"
    )

    out_cols = [
        config.date_col,
        "home_team",
        "away_team",
        "pick_type",
        "pick_side",
        "confidence",
        "reason",
    ]
    out_cols = [c for c in out_cols if c in picks.columns]
    audit = {
        "n_games": n_games,
        "n_candidates": int(len(candidates)),
        "n_picks": int(len(picks)),
        "policy": {
            "prob_floor": config.prob_floor,
            "max_picks_per_day": config.max_picks_per_day,
            "min_games_for_picks": config.min_games_for_picks,
        },
        "calibration_present": False,
        "note": "Fallback conservative picks used (Commit-3 module import not found).",
    }
    return picks[out_cols].copy(), audit


# -----------------------------
# File discovery helpers
# -----------------------------

def _list_prediction_files(pred_dir: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for p in sorted(pred_dir.glob("predictions_*.csv")):
        m = PRED_RE.match(p.name)
        if not m:
            continue
        out.append((m.group(1), p))
    return out


# -----------------------------
# Main runner
# -----------------------------

def run_historical_picks(
    *,
    pred_dir: Path,
    out_dir: Path,
    start: Optional[str],
    end: Optional[str],
    overwrite: bool,
    config: PicksRunnerConfig,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = out_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    files = _list_prediction_files(pred_dir)
    if not files:
        raise RuntimeError(f"[historical_picks] No prediction files found in {pred_dir} matching predictions_YYYY-MM-DD.csv")

    # Filter by date range (lexicographic works for YYYY-MM-DD)
    if start:
        files = [(d, p) for (d, p) in files if d >= start]
    if end:
        files = [(d, p) for (d, p) in files if d <= end]

    if not files:
        raise RuntimeError("[historical_picks] No prediction files remain after date filtering.")

    # Prefer existing conservative picks function if present
    commit3_fn = _try_import_commit3_picks()

    n_days = 0
    n_written = 0
    n_skipped_exists = 0
    n_skipped_empty = 0
    n_errors = 0
    errors: List[Dict] = []

    for game_date, pred_path in files:
        n_days += 1
        picks_path = out_dir / f"picks_{game_date}.csv"
        audit_path = audits_dir / f"picks_{game_date}_audit.json"

        if picks_path.exists() and not overwrite:
            n_skipped_exists += 1
            continue

        try:
            preds = pd.read_csv(pred_path)
            if preds.empty:
                n_skipped_empty += 1
                audit_path.write_text(
                    json.dumps(
                        {"game_date": game_date, "status": "skipped_empty_predictions", "pred_file": pred_path.name},
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                continue

            # Ensure game_date exists inside preds for downstream join/backtest
            if config.date_col not in preds.columns:
                preds[config.date_col] = game_date

            if commit3_fn is not None:
                # Try calling your existing function with flexible signature
                try:
                    picks, audit = commit3_fn(
                        preds,
                        prob_floor=config.prob_floor,
                        max_picks_per_day=config.max_picks_per_day,
                        min_games_for_picks=config.min_games_for_picks,
                        require_calibration_keep=config.require_calibration_keep,
                        max_abs_gap=config.max_abs_gap,
                    )
                except TypeError:
                    # Fallback if signature differs
                    picks, audit = commit3_fn(preds)
            else:
                picks, audit = _fallback_generate_conservative_picks(preds, config=config)

            # Write outputs (even if empty, for determinism and traceability)
            picks.to_csv(picks_path, index=False)
            audit_out = {
                "game_date": game_date,
                "pred_file": pred_path.name,
                "picks_file": picks_path.name,
                **(audit if isinstance(audit, dict) else {"audit": str(audit)}),
            }
            audit_path.write_text(json.dumps(audit_out, indent=2, sort_keys=True), encoding="utf-8")
            n_written += 1

            logger.info("[historical_picks] Wrote %s (%d rows)", picks_path.as_posix(), len(picks))

        except Exception as e:
            n_errors += 1
            err = {"game_date": game_date, "pred_file": pred_path.name, "error": repr(e)}
            errors.append(err)
            logger.exception("[historical_picks] error for %s: %s", game_date, e)
            # still write an audit stub so we don't silently drop
            audit_path.write_text(json.dumps({"game_date": game_date, "status": "error", **err}, indent=2, sort_keys=True), encoding="utf-8")

    runner_audit = {
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "start": start,
        "end": end,
        "overwrite": overwrite,
        "n_days_considered": n_days,
        "n_days_written": n_written,
        "n_skipped_exists": n_skipped_exists,
        "n_skipped_empty_predictions": n_skipped_empty,
        "n_errors": n_errors,
        "errors": errors[:50],  # cap
        "policy": {
            "prob_floor": config.prob_floor,
            "max_picks_per_day": config.max_picks_per_day,
            "min_games_for_picks": config.min_games_for_picks,
            "require_calibration_keep": config.require_calibration_keep,
            "max_abs_gap": config.max_abs_gap,
        },
        "note": "Additive historical picks generation; does not modify prediction artifacts.",
    }

    (out_dir / "historical_picks_runner_audit.json").write_text(
        json.dumps(runner_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return runner_audit


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="Generate picks_YYYY-MM-DD.csv across a date range from predictions_YYYY-MM-DD.csv.")
    ap.add_argument("--pred-dir", default="outputs", help="Directory containing predictions_YYYY-MM-DD.csv files")
    ap.add_argument("--out-dir", default="outputs", help="Directory to write picks_YYYY-MM-DD.csv files")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--overwrite", action="store_true", default=False)

    # Conservative policy knobs (Commit-3 defaults)
    ap.add_argument("--prob-floor", type=float, default=0.62)
    ap.add_argument("--max-picks-per-day", type=int, default=3)
    ap.add_argument("--min-games-for-picks", type=int, default=2)
    ap.add_argument("--require-calibration-keep", action="store_true", default=False)
    ap.add_argument("--max-abs-gap", type=float, default=0.08)

    args = ap.parse_args()

    audit = run_historical_picks(
        pred_dir=Path(args.pred_dir),
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
        overwrite=bool(args.overwrite),
        config=PicksRunnerConfig(
            prob_floor=float(args.prob_floor),
            max_picks_per_day=int(args.max_picks_per_day),
            min_games_for_picks=int(args.min_games_for_picks),
            require_calibration_keep=bool(args.require_calibration_keep),
            max_abs_gap=float(args.max_abs_gap),
        ),
    )
    print("[historical_picks] wrote outputs/historical_picks_runner_audit.json")
    print(f"[historical_picks] days_written={audit['n_days_written']} errors={audit['n_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
