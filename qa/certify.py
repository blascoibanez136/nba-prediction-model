#!/usr/bin/env python3
"""
QA certify script (Commit 2).

Runs:
  1) historical_prediction_runner to generate daily predictions
  2) backtest to join predictions with history + compute basic metrics

Usability goals:
- If you run `python qa/certify.py` from repo root, it should "just work" with sensible defaults.
- You can override paths/ranges via flags.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"[certify] $ {' '.join(cmd)}")
    subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ".")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/history/games_2019_2024.csv", help="Historical games CSV")
    ap.add_argument("--start", default="2023-10-24", help="YYYY-MM-DD")
    ap.add_argument("--end", default="2024-04-14", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--pred-dir", default="outputs", help="Where historical runner writes prediction files")
    ap.add_argument("--snapshot-dir", default="data/_snapshots", help="Odds snapshots directory (close_YYYYMMDD.csv)")
    ap.add_argument("--apply-market", action="store_true", help="Apply market close snapshots to prediction files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--pattern", default="predictions_*.csv", help="Glob pattern for pred files")
    ap.add_argument("--prob-col", default="home_win_prob")
    ap.add_argument("--spread-col", default="fair_spread")
    ap.add_argument("--total-col", default="fair_total")
    args = ap.parse_args()

    # 1) Historical predictions
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.historical_prediction_runner",
        "--history",
        args.history,
        "--start",
        args.start,
        "--end",
        args.end,
        "--out-dir",
        args.pred_dir,
        "--snapshot-dir",
        args.snapshot_dir,
        *(["--apply-market"] if args.apply_market else []),
        *(["--overwrite"] if args.overwrite else []),
    ])

    # 2) Backtest join + metrics
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.backtest",
        "--pred-dir",
        args.pred_dir,
        "--pattern",
        args.pattern,
        "--history",
        args.history,
        "--start",
        args.start,
        "--end",
        args.end,
        "--prob-col",
        args.prob_col,
        "--spread-col",
        args.spread_col,
        "--total-col",
        args.total_col,
    ])

    print("[certify] ✅ Done")


if __name__ == "__main__":
    main()
