from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def _hard_fail(msg: str) -> None:
    raise RuntimeError(f"[backtest] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--prob-col", required=True)
    parser.add_argument("--spread-col", required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        _hard_fail(f"Prediction dir not found: {pred_dir}")

    preds = sorted(pred_dir.glob("predictions_*.csv"))
    if not preds:
        _hard_fail("No prediction files found.")

    df_pred = pd.concat((pd.read_csv(p) for p in preds), ignore_index=True)
    df_res = pd.read_csv(args.results, parse_dates=["game_date"])

    df = df_pred.merge(
        df_res,
        on=["game_id"],
        how="inner",
        validate="many_to_one",
    )

    audit = {
        "pred_files": len(preds),
        "rows_joined": len(df),
        "date_range": [args.start, args.end],
    }

    audit_path = OUTPUTS_DIR / "backtest_join_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"[backtest] wrote {audit_path}")

    if df.empty:
        _hard_fail("Joined dataframe empty after merge.")


if __name__ == "__main__":
    main()
