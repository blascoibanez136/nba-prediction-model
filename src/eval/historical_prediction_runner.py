from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.model.predict import predict_games


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT / "models"
SNAPSHOT_DIR = REPO_ROOT / "data" / "_snapshots"


def _hard_fail(msg: str):
    raise RuntimeError(f"[historical] {msg}")


def _locate_models() -> Dict[str, Path]:
    required = {
        "win_model": "win_model.pkl",
        "spread_model": "spread_model.pkl",
        "total_model": "total_model.pkl",
        "teams": "team_index.json",
    }

    found = {}
    for key, fname in required.items():
        path = MODELS_DIR / fname
        if not path.exists():
            _hard_fail(f"Missing required model artifact: {path}")
        found[key] = path

    return found


def _load_models() -> Tuple[dict, object, object, object]:
    paths = _locate_models()

    import joblib

    teams = json.loads(paths["teams"].read_text())
    win_model = joblib.load(paths["win_model"])
    spread_model = joblib.load(paths["spread_model"])
    total_model = joblib.load(paths["total_model"])

    print("[historical] loaded models (teams + win/spread/total)")
    return teams, win_model, spread_model, total_model


def _run_day(
    df_day: pd.DataFrame,
    teams: dict,
    win_model,
    spread_model,
    total_model,
    apply_market: bool,
) -> pd.DataFrame:
    return predict_games(
        df_day,
        win_model=win_model,
        spread_model=spread_model,
        total_model=total_model,
        teams=teams,
        odds_dir=SNAPSHOT_DIR if apply_market else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    teams, win_model, spread_model, total_model = _load_models()

    df = pd.read_csv(args.history, parse_dates=["game_date"])
    mask = (df["game_date"] >= args.start) & (df["game_date"] <= args.end)
    df = df.loc[mask].copy()

    audit = {
        "date_range": [args.start, args.end],
        "apply_market": args.apply_market,
        "models_dir": str(MODELS_DIR),
        "snapshots_dir": str(SNAPSHOT_DIR),
        "prediction_files": [],
    }

    for game_date, df_day in df.groupby("game_date"):
        out = OUTPUTS_DIR / f"predictions_{game_date.date()}.csv"
        if out.exists() and not args.overwrite:
            continue

        df_pred = _run_day(
            df_day,
            teams,
            win_model,
            spread_model,
            total_model,
            apply_market=args.apply_market,
        )

        df_pred.to_csv(out, index=False)
        audit["prediction_files"].append(out.name)

        print(f"[historical] wrote {out} ({len(df_pred)} rows)")

    audit_path = OUTPUTS_DIR / "historical_prediction_runner_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[historical] wrote {audit_path}")


if __name__ == "__main__":
    main()
