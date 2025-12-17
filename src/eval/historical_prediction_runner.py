from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
SNAPSHOT_DIR = REPO_ROOT / "data" / "_snapshots"


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _info(msg: str) -> None:
    print(f"[historical] {msg}")


def _warn(msg: str) -> None:
    print(f"[historical][WARNING] {msg}")


def _load_models() -> Tuple[Any, Any, Any, Any]:
    """
    Keep behavior aligned with your current runner:
    - loads teams mapping + win/spread/total models from artifacts/models.
    """
    # NOTE: These imports are intentionally inside the function to keep the script import-safe.
    from src.model.load_models import load_models  # type: ignore

    teams, win_model, spread_model, total_model = load_models()
    return teams, win_model, spread_model, total_model


def _run_predict_for_day(df_day: pd.DataFrame, teams: Any, win_model: Any, spread_model: Any, total_model: Any) -> pd.DataFrame:
    from src.model.predict import predict_games  # type: ignore

    out = predict_games(df_day, teams=teams, win_model=win_model, spread_model=spread_model, total_model=total_model)
    return out


def _try_apply_market(df_pred: pd.DataFrame, game_date: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply market ensemble if snapshot exists, otherwise return original df.
    """
    diag: Dict[str, Any] = {"attempted": True, "applied": False, "reason": None}

    # You’re using CLOSE snapshots in your logs; keep that convention.
    # If your snapshot file naming differs, adjust here once and keep consistent.
    snap = SNAPSHOT_DIR / f"odds_snapshot_close_{game_date}.csv"
    if not snap.exists():
        diag["reason"] = f"no_close_snapshot:{snap.name}"
        return df_pred, diag

    try:
        from src.model.market_ensemble import apply_market_ensemble  # type: ignore

        df_out = apply_market_ensemble(df_pred, snapshot_path=str(snap))
        diag["applied"] = True
        return df_out, diag
    except Exception as e:
        diag["reason"] = f"market_apply_error:{type(e).__name__}"
        return df_pred, diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Historical prediction runner with audit output.")
    ap.add_argument("--history", required=True, help="Historical games CSV (e.g., data/history/games_2019_2024.csv)")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--apply-market", action="store_true", help="Attempt to apply market ensemble when snapshots exist")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing daily prediction files")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # audit container
    audit: Dict[str, Any] = {
        "status": "ok",
        "window": {"start": args.start, "end": args.end},
        "inputs": {"history": args.history, "apply_market": bool(args.apply_market), "overwrite": bool(args.overwrite)},
        "model_files": {},
        "days": {
            "processed": 0,
            "written": 0,
            "skipped_existing": 0,
            "skipped_no_games": 0,
        },
        "market": {
            "attempted_days": 0,
            "applied_days": 0,
            "skipped_days": 0,
            "skip_reasons": {},
        },
        "unseen_teams": {
            "days_with_unseen": 0,
            "max_unseen_rate": 0.0,
        },
        "outputs_sample": [],
    }

    # load history
    hist = pd.read_csv(args.history)
    if "game_date" not in hist.columns:
        raise RuntimeError("[historical] history CSV missing required column: game_date")
    hist["game_date"] = hist["game_date"].astype(str)

    # load models
    teams, win_model, spread_model, total_model = _load_models()
    _info("loaded models: teams=30 win=win_model.pkl spread=spread_model.pkl total=total_model.pkl")

    start_dt = _parse_date(args.start)
    end_dt = _parse_date(args.end)

    d = start_dt
    while d <= end_dt:
        game_date = _date_str(d)
        audit["days"]["processed"] += 1

        out_path = OUTPUTS_DIR / f"predictions_{game_date}.csv"
        if out_path.exists() and not args.overwrite:
            audit["days"]["skipped_existing"] += 1
            d += timedelta(days=1)
            continue

        df_day = hist[hist["game_date"] == game_date].copy()
        if df_day.empty:
            audit["days"]["skipped_no_games"] += 1
            d += timedelta(days=1)
            continue

        # predict
        df_pred = _run_predict_for_day(df_day, teams, win_model, spread_model, total_model)

        # unseen team diagnostics (mirrors your warnings)
        unseen_rate = float(df_pred.get("unseen_team_rate", pd.Series([0.0])).max()) if len(df_pred) else 0.0
        if unseen_rate > 0:
            audit["unseen_teams"]["days_with_unseen"] += 1
            audit["unseen_teams"]["max_unseen_rate"] = float(max(audit["unseen_teams"]["max_unseen_rate"], unseen_rate))

        # market ensemble (optional)
        if args.apply_market:
            audit["market"]["attempted_days"] += 1
            df_pred2, diag = _try_apply_market(df_pred, game_date)
            if diag.get("applied"):
                audit["market"]["applied_days"] += 1
                df_pred = df_pred2
            else:
                audit["market"]["skipped_days"] += 1
                r = diag.get("reason") or "unknown"
                audit["market"]["skip_reasons"][r] = int(audit["market"]["skip_reasons"].get(r, 0) + 1)
                # keep your existing log tone
                if r.startswith("no_close_snapshot"):
                    _warn(f"No CLOSE odds snapshot found for {game_date} in data/_snapshots — skipping market ensemble")
                else:
                    _warn(f"Market ensemble skipped for {game_date}: {r}")

        # write
        df_pred.to_csv(out_path, index=False)
        audit["days"]["written"] += 1

        if len(audit["outputs_sample"]) < 15:
            audit["outputs_sample"].append({"date": game_date, "path": str(out_path), "rows": int(len(df_pred))})

        _info(f"Wrote {out_path} ({len(df_pred)} rows)")

        d += timedelta(days=1)

    # finalize audit
    audit_path = OUTPUTS_DIR / "historical_prediction_runner_audit.json"
    _write_json(audit_path, audit)
    _info("wrote outputs/historical_prediction_runner_audit.json")


if __name__ == "__main__":
    main()
