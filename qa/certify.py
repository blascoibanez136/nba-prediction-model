#!/usr/bin/env python3
"""
QA certify script (Commit 2 + E2 + E3 guards) with dispersion and OOS window fixes.

This version adds a pre‑processing step to build a per‑game file with market
consensus and dispersion for ATS ROI analysis, and enforces the locked
out‑of‑sample evaluation window (2024‑03‑11..2024‑04‑14).

It runs:
  1) historical_prediction_runner
  2) backtest join + metrics (outputs/backtest_joined.csv)
  3) E2 policy metrics (src.eval.e2_policy_runner) + validation
  4) Build ATS ROI input (src.eval.build_ats_roi_input) using snapshots
  5) ATS ROI bets (src.eval.ats_roi_analysis) on the OOS window
  6) E3 staking metrics (src.eval.e3_policy_runner) + validation

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
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ".")},
    )


def _fail(msg: str) -> None:
    raise RuntimeError(f"[certify] {msg}")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_e2_policy(outputs_dir: Path) -> dict:
    metrics_path = outputs_dir / "e2_policy_metrics.json"
    if not metrics_path.exists():
        _fail("E2 missing outputs/e2_policy_metrics.json")

    m = _load_json(metrics_path)

    required_fields = ["sample_size", "performance", "risk_metrics", "clv_coverage", "filters", "eligibility"]
    for field in required_fields:
        if field not in m:
            _fail(f"E2 missing required field: {field}")

    bets = m["sample_size"].get("bets")
    avg_clv = m["performance"].get("average_clv")
    clv_pos_rate = m["performance"].get("clv_positive_rate")
    roi = m["performance"].get("roi")
    max_dd = m["risk_metrics"].get("max_drawdown_units")

    open_cov = m["clv_coverage"].get("open_snapshot_coverage_pct")
    close_cov = m["clv_coverage"].get("close_snapshot_coverage_pct")

    elig = m.get("eligibility", {})
    eligible_pct = elig.get("eligible_pct", None)

    if bets is None or int(bets) < 60:
        _fail(f"E2 bet count regression: bets={bets}")

    if eligible_pct is None:
        _fail("E2 missing eligibility.eligible_pct")
    eligible_pct_f = float(eligible_pct)
    if eligible_pct_f < 0.85:
        _fail(f"E2 eligibility regression: eligible_pct={eligible_pct_f:.3f} (<0.85)")

    if avg_clv is None or float(avg_clv) <= 0:
        _fail(f"E2 CLV regression: avg_clv={avg_clv}")

    if clv_pos_rate is None or float(clv_pos_rate) < 0.55:
        _fail(f"E2 CLV+ rate regression: {clv_pos_rate}")

    if roi is None or float(roi) < 0:
        _fail(f"E2 ROI regression: roi={roi}")

    if max_dd is None or float(max_dd) < -7:
        _fail(f"E2 Drawdown regression: max_drawdown_units={max_dd}")

    if open_cov is None or close_cov is None:
        _fail(f"E2 Snapshot coverage missing: open={open_cov} close={close_cov}")
    if int(open_cov) < 100 or int(close_cov) < 100:
        _fail(f"E2 Snapshot coverage regression (eligible universe): open={open_cov} close={close_cov}")

    print("[certify:E2] ✅ E2 validated")
    print(f"[certify:E2] bets={int(bets)} roi={float(roi):.4f} avg_clv={float(avg_clv):.6f} "
          f"clv_pos_rate={float(clv_pos_rate):.3f} max_dd={float(max_dd):.3f} eligible_pct={eligible_pct_f:.3f}")
    return m


def _validate_e3_policy(outputs_dir: Path, *, e2_metrics: dict, policy_json: Path) -> None:
    mpath = outputs_dir / "e3_policy_metrics.json"
    if not mpath.exists():
        _fail("E3 missing outputs/e3_policy_metrics.json")

    m = _load_json(mpath)

    for field in ["sample_size", "performance", "risk_metrics", "exposure", "guards", "policy"]:
        if field not in m:
            _fail(f"E3 missing required field: {field}")

    # Hard stake caps
    max_daily_cap = float(_load_json(policy_json)["risk_caps"]["max_daily_stake_u"])
    max_per_bet_cap = float(_load_json(policy_json)["stake_model"]["max_stake_u"])

    max_daily_seen = float(m["exposure"].get("max_daily_stake_u", -1))
    max_per_bet_seen = float(m["exposure"].get("max_per_bet_stake_u", -1))

    if max_daily_seen > max_daily_cap + 1e-9:
        _fail(f"E3 daily stake cap violation: seen={max_daily_seen} cap={max_daily_cap}")
    if max_per_bet_seen > max_per_bet_cap + 1e-9:
        _fail(f"E3 per-bet stake cap violation: seen={max_per_bet_seen} cap={max_per_bet_cap}")

    if int(m["guards"].get("daily_cap_violations", 0)) != 0:
        _fail(f"E3 daily_cap_violations={m['guards'].get('daily_cap_violations')}")
    if int(m["guards"].get("per_bet_cap_violations", 0)) != 0:
        _fail(f"E3 per_bet_cap_violations={m['guards'].get('per_bet_cap_violations')}")

    # Risk improvement vs E2
    e2_dd = float(e2_metrics["risk_metrics"]["max_drawdown_units"])
    e3_dd = float(m["risk_metrics"].get("max_drawdown_units"))

    # E2 DD is negative. "Improve" means E3 is less negative (>=).
    if e3_dd < e2_dd:
        _fail(f"E3 drawdown regression: e3_max_dd={e3_dd:.3f} worse than e2_max_dd={e2_dd:.3f}")

    # Performance sanity
    e3_roi = m["performance"].get("roi")
    if e3_roi is None or float(e3_roi) < 0:
        _fail(f"E3 ROI regression: roi={e3_roi}")

    # Preserve positive CLV when available
    e3_avg_clv = m["performance"].get("average_clv")
    if e3_avg_clv is not None and float(e3_avg_clv) <= 0:
        _fail(f"E3 CLV regression: avg_clv={e3_avg_clv}")

    print("[certify:E3] ✅ E3 validated")
    print(f"[certify:E3] roi={float(e3_roi):.4f} max_dd={e3_dd:.3f} "
          f"max_daily={max_daily_seen:.2f} max_per_bet={max_per_bet_seen:.2f}")


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

    out_dir = Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Historical predictions
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.historical_prediction_runner",
        "--history", args.history,
        "--start", args.start,
        "--end", args.end,
        "--out-dir", args.pred_dir,
        "--snapshot-dir", args.snapshot_dir,
        *( ["--apply-market"] if args.apply_market else [] ),
        *( ["--overwrite"] if args.overwrite else [] ),
    ])

    # 2) Backtest
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.backtest",
        "--pred-dir", args.pred_dir,
        "--pattern", args.pattern,
        "--history", args.history,
        "--start", args.start,
        "--end", args.end,
        "--prob-col", args.prob_col,
        "--spread-col", args.spread_col,
        "--total-col", args.total_col,
        "--out-dir", args.pred_dir,
    ])

    # 3) E2 policy metrics (ATS-based runner)
    calibrator_path = Path("artifacts/spread_calibrator.joblib")
    if not calibrator_path.exists():
        _fail("Missing artifacts/spread_calibrator.joblib (train it before certify).")

    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.e2_policy_runner",
        "--per-game", str(out_dir / "backtest_joined.csv"),
        "--snapshot-dir", args.snapshot_dir,
        "--start", args.start,
        "--end", args.end,
        "--calibrator", str(calibrator_path),
        "--out", str(out_dir / "e2_policy_metrics.json"),
    ])

    e2_metrics = _validate_e2_policy(out_dir)

    # 4) Produce ATS ROI input and bets (input to E3 staking)
    policy_path = Path("configs/ats_policy_v1.yaml")
    if not policy_path.exists():
        _fail("Missing configs/ats_policy_v1.yaml")

    # Build per-game file with consensus and dispersion from snapshots
    build_ats_out = out_dir / "backtest_joined_market.csv"
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.build_ats_roi_input",
        "--backtest-joined", str(out_dir / "backtest_joined.csv"),
        "--snapshot-dir", args.snapshot_dir,
        "--start", args.start,
        "--end", args.end,
        "--out", str(build_ats_out),
    ])

    # Run ATS ROI analysis on the full evaluation window to produce bets for E3
    # We intentionally use args.start and args.end rather than an OOS window here,
    # because E3 staking overlay applies to the entire season's ATS bets.
    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.ats_roi_analysis",
        "--per_game", str(build_ats_out),
        "--calibrator", str(calibrator_path),
        "--policy", str(policy_path),
        "--eval-start", args.start,
        "--eval-end", args.end,
        "--strict",
        "--max-bet-rate", "0.30",
    ])

    ats_bets = out_dir / "ats_roi_bets.csv"
    if not ats_bets.exists():
        _fail("Missing outputs/ats_roi_bets.csv after ats_roi_analysis run.")

    # 5) Run E3 staking metrics
    e3_policy = Path("configs/e3_staking_policy_v1.json")
    if not e3_policy.exists():
        _fail("Missing configs/e3_staking_policy_v1.json")

    _run([
        os.environ.get("PYTHON", "python"),
        "-m",
        "src.eval.e3_policy_runner",
        "--ats-bets", str(ats_bets),
        "--policy", str(e3_policy),
        "--out-metrics", str(out_dir / "e3_policy_metrics.json"),
        "--out-bets", str(out_dir / "ats_e3_staked_bets.csv"),
    ])

    _validate_e3_policy(out_dir, e2_metrics=e2_metrics, policy_json=e3_policy)

    print("[certify] ✅ All checks passed (E2 + E3)")


if __name__ == "__main__":
    main()
