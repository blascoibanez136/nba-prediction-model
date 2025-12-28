#!/usr/bin/env python3
"""
QA certify script (Commit 2 + E2 policy guard).

Runs:
  1) historical_prediction_runner
  2) backtest join + metrics
  3) generate E2 policy metrics (e2_policy_runner)
  4) validate E2 policy metrics (CLV / ROI / risk)

Any regression fails hard.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"[certify] $ {' '.join(cmd)}")
    subprocess.check_call(
        cmd,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ".")}
    )


def _fail(msg: str) -> None:
    raise RuntimeError(f"[certify:E2] {msg}")


def _validate_e2_policy(outputs_dir: Path) -> None:
    """
    Hard regression checks for Phase E2.
    Eligibility-aware:

    - Real-world snapshots can have partial ML coverage.
    - e2_policy_runner computes metrics ONLY on eligible rows (full open+close ML availability).
    - Coverage must be 100% on the eligible universe.
    - Eligible universe must be large enough (eligible_pct guard).
    """
    metrics_path = outputs_dir / "e2_policy_metrics.json"

    if not metrics_path.exists():
        _fail("Missing outputs/e2_policy_metrics.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # --- Required fields ---
    required_fields = [
        "sample_size",
        "performance",
        "risk_metrics",
        "clv_coverage",
        "filters",
    ]
    for field in required_fields:
        if field not in m:
            _fail(f"Missing required field: {field}")

    bets = m["sample_size"].get("bets")
    avg_clv = m["performance"].get("average_clv")
    clv_pos_rate = m["performance"].get("clv_positive_rate")
    roi = m["performance"].get("roi")
    max_dd = m["risk_metrics"].get("max_drawdown_units")

    open_cov = m["clv_coverage"].get("open_snapshot_coverage_pct")
    close_cov = m["clv_coverage"].get("close_snapshot_coverage_pct")

    # Eligibility transparency (new)
    eligibility = m.get("eligibility", {}) if isinstance(m.get("eligibility", {}), dict) else {}
    eligible_pct = eligibility.get("eligible_pct", None)

    # --- Hard guards (tuneable later) ---
    if bets is None or int(bets) < 60:
        _fail(f"Bet count regression: bets={bets}")

    # Ensure we didn't silently pass on a tiny eligible subset
    if eligible_pct is None:
        _fail("Missing eligibility.eligible_pct (required for eligibility-aware certification).")
    try:
        eligible_pct_f = float(eligible_pct)
    except Exception:
        _fail(f"Invalid eligibility.eligible_pct: {eligible_pct}")
    if eligible_pct_f < 0.85:
        _fail(f"Eligibility regression: eligible_pct={eligible_pct_f:.3f} (<0.85)")

    if avg_clv is None or float(avg_clv) <= 0:
        _fail(f"CLV regression: avg_clv={avg_clv}")

    if clv_pos_rate is None or float(clv_pos_rate) < 0.55:
        _fail(f"CLV+ rate regression: {clv_pos_rate}")

    if roi is None or float(roi) < 0:
        _fail(f"ROI regression: roi={roi}")

    if max_dd is None or float(max_dd) < -7:
        _fail(f"Drawdown regression: max_drawdown_units={max_dd}")

    # Coverage must be 100% ON ELIGIBLE UNIVERSE (not all games)
    if open_cov is None or close_cov is None:
        _fail(f"Snapshot coverage missing: open={open_cov} close={close_cov}")
    if int(open_cov) < 100 or int(close_cov) < 100:
        _fail(
            f"Snapshot coverage regression (eligible universe): "
            f"open={open_cov} close={close_cov}"
        )

    print("[certify:E2] ✅ E2 policy metrics validated")
    print(
        f"[certify:E2] bets={bets} roi={float(roi):.4f} avg_clv={float(avg_clv):.6f} "
        f"clv_pos_rate={float(clv_pos_rate):.3f} max_dd={float(max_dd):.3f} eligible_pct={eligible_pct_f:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/history/games_2019_2024.csv")
    ap.add_argument("--start", default="2023-10-24")
    ap.add_argument("--end", default="2024-04-14")
    ap.add_argument("--pred-dir", default="outputs")
    ap.add_argument("--snapshot-dir", default="data/_snapshots")
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pattern", default="predictions_*.csv")
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

    # 2) Backtest (writes outputs/backtest_joined.csv)
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
        "--out-dir",
        args.pred_dir,
    ])

    # 3) Generate E2 policy metrics
    # IMPORTANT: certify produces backtest_joined.csv by default.
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.e2_policy_runner",
        "--per-game",
        str(Path(args.pred_dir) / "backtest_joined.csv"),
        "--snapshot-dir",
        args.snapshot_dir,
        "--start",
        args.start,
        "--end",
        args.end,
        "--out",
        str(Path(args.pred_dir) / "e2_policy_metrics.json"),
    ])

    # 4) Validate E2 policy metrics
    _validate_e2_policy(Path(args.pred_dir))

    print("[certify] ✅ All checks passed")


if __name__ == "__main__":
    main()
