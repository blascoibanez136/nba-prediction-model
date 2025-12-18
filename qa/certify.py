from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

DEFAULT_HISTORY = REPO_ROOT / "data" / "history" / "games_2019_2024.csv"
DEFAULT_START = "2023-10-24"
DEFAULT_END = "2024-04-14"


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _run(cmd: list[str]) -> None:
    print(f"[certify] $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _require_file(path: Path, name: str) -> CheckResult:
    if path.exists() and path.stat().st_size > 0:
        return CheckResult(name=name, ok=True, details=str(path))
    return CheckResult(name=name, ok=False, details=f"missing_or_empty:{path}")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="QA certification: run historical predictions + backtest and assert audits exist.")
    p.add_argument("--history", type=str, default=str(DEFAULT_HISTORY))
    p.add_argument("--start", type=str, default=DEFAULT_START)
    p.add_argument("--end", type=str, default=DEFAULT_END)
    p.add_argument("--apply-market", action="store_true", help="Attempt market overlay using CLOSE snapshots during historical run.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite prediction files during historical run.")
    p.add_argument("--skip-historical", action="store_true")
    p.add_argument("--skip-backtest", action="store_true")
    args = p.parse_args(argv)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    checks: list[CheckResult] = []

    if not args.skip_historical:
        _run([
            sys.executable, "-m", "src.eval.historical_prediction_runner",
            "--history", args.history,
            "--start", args.start,
            "--end", args.end,
            * (["--apply-market"] if args.apply_market else []),
            * (["--overwrite"] if args.overwrite else []),
        ])
        checks.append(_require_file(OUTPUTS / "historical_prediction_runner_audit.json", "historical_audit"))

    if not args.skip_backtest:
        prob_col = "home_win_prob"
        spread_col = "fair_spread"
        total_col = "fair_total"

        sample = next(iter(sorted(OUTPUTS.glob("predictions_*.csv"))), None)
        if sample is not None:
            import pandas as pd
            df = pd.read_csv(sample, nrows=1)
            if "home_win_prob_market" in df.columns:
                prob_col = "home_win_prob_market"
            if "fair_spread_market" in df.columns:
                spread_col = "fair_spread_market"
            if "fair_total_market" in df.columns:
                total_col = "fair_total_market"

        _run([
            sys.executable, "-m", "src.eval.backtest",
            "--pred-dir", str(OUTPUTS),
            "--history", args.history,
            "--start", args.start,
            "--end", args.end,
            "--prob-col", prob_col,
            "--spread-col", spread_col,
            "--total-col", total_col,
        ])
        checks.append(_require_file(OUTPUTS / "backtest_join_audit.json", "backtest_audit"))
        checks.append(_require_file(OUTPUTS / "backtest_metrics.json", "backtest_metrics"))

    ok = all(c.ok for c in checks) if checks else True
    summary = {
        "ok": ok,
        "checks": [c.__dict__ for c in checks],
        "history": args.history,
        "start": args.start,
        "end": args.end,
        "apply_market": bool(args.apply_market),
    }
    (OUTPUTS / "certify_summary.json").write_text(json.dumps(summary, indent=2))

    print("[certify] Results:")
    for c in checks:
        print(f"  - {c.name}: {'OK' if c.ok else 'FAIL'} ({c.details})")

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
