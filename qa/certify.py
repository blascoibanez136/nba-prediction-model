"""QA entrypoint: run historical prediction runner, then backtest.

Commit-2 goal:
- Provide a single command to certify the pipeline works end-to-end in a fresh environment.
- Historical runner must produce per-day CSVs.
- Backtest must join those CSVs to history/results and emit metrics without crashing.

Note: Backtest is intentionally tolerant of missing score columns; it will still emit an audit + metrics
with status=missing_score_columns rather than failing the run.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"[certify] $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to games history CSV.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    py = "/usr/bin/python3"

    _run([
        py, "-m", "src.eval.historical_prediction_runner",
        "--history", args.history,
        "--start", args.start,
        "--end", args.end,
        *(["--apply-market"] if args.apply_market else []),
        *(["--overwrite"] if args.overwrite else []),
    ])

    _run([
        py, "-m", "src.eval.backtest",
        "--pred-dir", str(REPO_DIR / "outputs"),
        "--history", args.history,
        "--start", args.start,
        "--end", args.end,
        "--prob-col", "home_win_prob",
        "--spread-col", "fair_spread",
        "--total-col", "fair_total",
    ])


if __name__ == "__main__":
    main()
