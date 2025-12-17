from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Team normalizer (critical for merge-key contract) ----
# Prefer the canonical location used by ingest.
try:
    from src.ingest.team_normalizer import normalize_team_name  # type: ignore
except Exception:  # pragma: no cover
    # Fallback (keeps script runnable even if paths change again)
    def normalize_team_name(x: Any) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        s = str(x).strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _merge_key(home: str, away: str, game_date: str) -> str:
    return f"{home.lower()}__{away.lower()}__{game_date}"


def _warn(msg: str) -> None:
    print(f"[backtest][WARN] {msg}")


def _info(msg: str) -> None:
    print(f"[backtest] {msg}")


def _hard_fail(msg: str) -> None:
    raise RuntimeError(f"[backtest] {msg}")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "game_date" not in df.columns:
        _hard_fail("results file missing required column: game_date")
    # Normalize columns (some historical files vary)
    for col in ["home_team", "away_team"]:
        if col not in df.columns:
            _hard_fail(f"results file missing required column: {col}")
    return df


def _load_predictions(pred_files: List[Path]) -> pd.DataFrame:
    parts = []
    for p in pred_files:
        df = pd.read_csv(p)
        if "game_date" not in df.columns:
            # derive from filename if possible
            m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
            if m:
                df["game_date"] = m.group(1)
            else:
                _hard_fail(f"prediction file missing game_date and cannot infer date: {p}")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out


def _select_pred_files(pred_dir: Path, pattern: str, start: str, end: str) -> List[Path]:
    files = [Path(x) for x in glob.glob(str(pred_dir / pattern))]
    if not files:
        return []
    s0, s1 = _parse_date(start), _parse_date(end)
    keep = []
    for p in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if not m:
            continue
        d = _parse_date(m.group(1))
        if s0 <= d <= s1:
            keep.append(p)
    keep.sort()
    return keep


def _add_norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_team_norm"] = out["home_team"].apply(normalize_team_name)
    out["away_team_norm"] = out["away_team"].apply(normalize_team_name)
    out["merge_key"] = out.apply(lambda r: _merge_key(r["home_team_norm"], r["away_team_norm"], str(r["game_date"])), axis=1)
    return out


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _spread_errors(df: pd.DataFrame, spread_col: str) -> Dict[str, Any]:
    if spread_col not in df.columns:
        return {}
    if "home_score" not in df.columns or "away_score" not in df.columns:
        return {}
    # actual margin = home - away
    actual = (df["home_score"].astype(float) - df["away_score"].astype(float)).to_numpy()
    pred = df[spread_col].astype(float).to_numpy()
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse}


def _write_audit(
    out_dir: Path,
    audit: Dict[str, Any],
) -> None:
    _write_json(out_dir / "backtest_join_audit.json", audit)
    _info("wrote outputs/backtest_join_audit.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtesting engine with merge-coverage enforcement + join audit.")
    ap.add_argument("--pred-dir", required=True, help="Directory containing predictions_*.csv files.")
    ap.add_argument("--pattern", default="predictions_*_market.csv", help="Glob pattern for prediction files.")
    ap.add_argument("--results", required=True, help="Results CSV path (historical games).")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    ap.add_argument("--prob-col", default="home_win_prob", help="Probability column for win-prob metrics.")
    ap.add_argument("--spread-col", default="fair_spread", help="Spread column for error metrics.")
    ap.add_argument("--metrics-path", default=str(DEFAULT_OUT_DIR / "backtest_metrics.json"))
    ap.add_argument("--calib-path", default=str(DEFAULT_OUT_DIR / "backtest_calibration.csv"))
    ap.add_argument("--per-game-path", default=str(DEFAULT_OUT_DIR / "backtest_per_game.csv"))
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    results_csv = Path(args.results)
    out_dir = Path(args.metrics_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    pred_files = _select_pred_files(pred_dir, args.pattern, args.start, args.end)
    if not pred_files:
        audit = {
            "status": "fail",
            "reason": "no_prediction_files",
            "pred_dir": str(pred_dir),
            "pattern": args.pattern,
            "start": args.start,
            "end": args.end,
        }
        _write_audit(out_dir, audit)
        _hard_fail("No prediction files matched pattern/date window.")

    preds = _load_predictions(pred_files)
    results = _load_results(results_csv)

    # filter results to window
    s0, s1 = _parse_date(args.start), _parse_date(args.end)
    results = results.copy()
    results["game_date"] = results["game_date"].astype(str)
    results["_dt"] = results["game_date"].apply(_parse_date)
    results = results[(results["_dt"] >= s0) & (results["_dt"] <= s1)].drop(columns=["_dt"])

    preds = _add_norm_keys(preds)
    results = _add_norm_keys(results)

    # ---- merge + coverage ----
    merged = preds.merge(
        results,
        on="merge_key",
        how="inner",
        suffixes=("_pred", "_res"),
    )

    total_results_games = int(len(results))
    matched_games = int(len(merged))
    coverage = (matched_games / total_results_games) if total_results_games else 0.0

    # Coverage diagnostics
    results_keys = set(results["merge_key"].tolist())
    pred_keys = set(preds["merge_key"].tolist())
    missing_pred = sorted(list(results_keys - pred_keys))[:50]  # cap
    missing_res = sorted(list(pred_keys - results_keys))[:50]   # cap

    audit: Dict[str, Any] = {
        "status": "ok",
        "window": {"start": args.start, "end": args.end},
        "inputs": {
            "pred_dir": str(pred_dir),
            "pattern": args.pattern,
            "results_csv": str(results_csv),
            "pred_files": [str(p) for p in pred_files],
        },
        "counts": {
            "results_games": total_results_games,
            "pred_rows": int(len(preds)),
            "matched_games": matched_games,
        },
        "coverage": {
            "rate": float(round(coverage, 6)),
            "warn_lt": 0.95,
            "fail_lt": 0.80,
        },
        "missing_examples": {
            "missing_pred_keys_sample": missing_pred,
            "missing_result_keys_sample": missing_res,
        },
        "columns": {
            "prob_col": args.prob_col,
            "spread_col": args.spread_col,
        },
    }

    # Write audit BEFORE any gates (so failures still produce the audit file)
    _write_audit(out_dir, audit)

    if coverage < 0.80:
        _hard_fail(f"Merge coverage too low: {coverage:.3f} (<0.80 hard fail). See outputs/backtest_join_audit.json")
    if coverage < 0.95:
        _warn(f"Merge coverage low: {coverage:.3f} (<0.95 warn). See outputs/backtest_join_audit.json")

    # ---- metrics ----
    metrics: Dict[str, Any] = {
        "window": {"start": args.start, "end": args.end},
        "coverage": float(round(coverage, 6)),
        "matched_games": matched_games,
        "results_games": total_results_games,
    }

    # win-prob metrics
    if args.prob_col in merged.columns:
        # determine outcome from scores when present
        if "home_score" in merged.columns and "away_score" in merged.columns:
            y = (merged["home_score"].astype(float) > merged["away_score"].astype(float)).astype(int).to_numpy()
            p = merged[args.prob_col].astype(float).to_numpy()
            metrics["brier"] = _brier(y, p)
            metrics["log_loss"] = _logloss(y, p)
        else:
            _warn("home_score/away_score not present in merged frame; skipping brier/logloss.")
    else:
        _warn(f"prob-col '{args.prob_col}' not found; skipping brier/logloss.")

    # spread metrics
    se = _spread_errors(merged, args.spread_col)
    if se:
        metrics["spread_error"] = se

    # ---- write outputs ----
    metrics_path = Path(args.metrics_path)
    _write_json(metrics_path, metrics)
    _info(f"wrote: {metrics_path}")

    # per-game
    per_game_path = Path(args.per_game_path)
    merged.to_csv(per_game_path, index=False)
    _info(f"wrote: {per_game_path}")

    # calibration csv (lightweight)
    calib_path = Path(args.calib_path)
    if args.prob_col in merged.columns and "home_score" in merged.columns and "away_score" in merged.columns:
        tmp = merged[[args.prob_col, "home_score", "away_score"]].copy()
        tmp["y"] = (tmp["home_score"].astype(float) > tmp["away_score"].astype(float)).astype(int)
        tmp.to_csv(calib_path, index=False)
        _info(f"wrote: {calib_path}")
    else:
        _warn("Skipping calibration csv (missing prob-col or scores).")

    _info("DONE")


if __name__ == "__main__":
    main()
