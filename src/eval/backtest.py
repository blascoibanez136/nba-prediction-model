from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def _hard_fail(msg: str):
    raise RuntimeError(f"[backtest] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--prob-col", required=True)
    ap.add_argument("--spread-col", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        _hard_fail(f"Prediction directory not found: {pred_dir}")

    files = sorted(pred_dir.glob("predictions_*.csv"))
    if not files:
        _hard_fail("No prediction files found.")

    df_preds = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file"] = f.name
        df_preds.append(df)

    preds = pd.concat(df_preds, ignore_index=True)

    results = pd.read_csv(args.results, parse_dates=["game_date"])
    mask = (results["game_date"] >= args.start) & (results["game_date"] <= args.end)
    results = results.loc[mask]

    merged = preds.merge(
        results,
        on=["game_id"],
        how="left",
        suffixes=("", "_actual"),
    )

    audit = {
        "prediction_files": [f.name for f in files],
        "rows_predictions": len(preds),
        "rows_joined": len(merged),
        "date_range": [args.start, args.end],
    }

    audit_path = OUTPUTS_DIR / "backtest_join_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[backtest] wrote {audit_path}")


if __name__ == "__main__":
    main()
