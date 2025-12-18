from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# --- robust import (works with -m and script) ---
try:
    from src.model.predict import predict_games
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.model.predict import predict_games  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT / "models"
SNAPSHOTS_DIR = REPO_ROOT / "data" / "_snapshots"


def _load_models() -> Dict[str, Any]:
    required = {
        "win_model": MODELS_DIR / "win_model.pkl",
        "spread_model": MODELS_DIR / "spread_model.pkl",
        "total_model": MODELS_DIR / "total_model.pkl",
        "teams": MODELS_DIR / "team_index.json",
    }

    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing model artifacts: {missing}")

    import joblib

    return {
        "win_model": joblib.load(required["win_model"]),
        "spread_model": joblib.load(required["spread_model"]),
        "total_model": joblib.load(required["total_model"]),
        "teams": json.loads(required["teams"].read_text()),
    }


def _run_day(
    df_day: pd.DataFrame,
    models: Dict[str, Any],
    apply_market: bool,
) -> pd.DataFrame:
    # NOTE: predict_games() DOES NOT accept named model args
    return predict_games(
        df_day,
        models["teams"],
        models["win_model"],
        models["spread_model"],
        models["total_model"],
        apply_market=apply_market,
        snapshots_dir=SNAPSHOTS_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--apply-market", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    audit: Dict[str, Any] = {
        "models_dir": str(MODELS_DIR),
        "snapshots_dir": str(SNAPSHOTS_DIR),
        "written_files": [],
    }

    models = _load_models()
    print("[historical] loaded models (teams + win/spread/total)")

    df = pd.read_csv(args.history, parse_dates=["game_date"])
    mask = (df["game_date"] >= args.start) & (df["game_date"] <= args.end)
    df = df.loc[mask]

    for date, df_day in df.groupby(df["game_date"].dt.date):
        out = OUTPUTS_DIR / f"predictions_{date}.csv"
        if out.exists() and not args.overwrite:
            continue

        df_pred = _run_day(df_day, models, args.apply_market)
        df_pred.to_csv(out, index=False)
        audit["written_files"].append(str(out))
        print(f"[historical] wrote {out} ({len(df_pred)} rows)")

    audit_path = OUTPUTS_DIR / "historical_prediction_runner_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[historical] wrote {audit_path}")


if __name__ == "__main__":
    main()
