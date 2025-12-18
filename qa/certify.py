from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]

PER_GAME_FIXTURE = REPO_ROOT / "qa" / "fixtures" / "backtest_per_game.csv"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Locked windows (system of record)
TRAIN_START = "2023-10-24"
TRAIN_END = "2024-03-10"
EVAL_START = "2024-03-11"
EVAL_END = "2024-04-14"

REQUIRED_OUTPUTS = [
    OUTPUTS_DIR / "ats_roi_metrics.json",
    OUTPUTS_DIR / "totals_roi_metrics.json",
    OUTPUTS_DIR / "ml_roi_metrics.json",
]

# Optional “commit 2” artifacts (audits)
HISTORICAL_AUDIT = OUTPUTS_DIR / "historical_prediction_runner_audit.json"
BACKTEST_AUDIT = OUTPUTS_DIR / "backtest_join_audit.json"


def run(cmd: List[str]) -> None:
    print("\n[certify] ▶", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def ensure_paths() -> None:
    if not PER_GAME_FIXTURE.exists():
        raise RuntimeError(f"[certify] Missing fixture: {PER_GAME_FIXTURE}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def run_ats_regression() -> None:
    print("\n[certify] === ATS Policy v1 Regression ===")
    run([sys.executable, "qa/regression_ats_policy_v1.py"])


def run_totals_suite() -> None:
    print("\n[certify] === Totals v3 ===")

    total_cal = ARTIFACTS_DIR / "total_calibrator.joblib"

    run([
        sys.executable,
        "-m", "src.eval.train_total_calibrator",
        "--per-game", str(PER_GAME_FIXTURE),
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--out", str(total_cal),
    ])

    run([
        sys.executable,
        "-m", "src.eval.totals_roi_analysis",
        "--per-game", str(PER_GAME_FIXTURE),
        "--calibrator", str(total_cal),
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--max-bet-rate", "0.30",
    ])


def run_ml_suite() -> None:
    print("\n[certify] === ML Selector v3 ===")

    delta_cal = ARTIFACTS_DIR / "delta_calibrator.joblib"

    run([
        sys.executable,
        "-m", "src.eval.train_delta_calibrator",
        "--per-game", str(PER_GAME_FIXTURE),
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--out", str(delta_cal),
    ])

    # FIX: use --delta-calibrator (not --calibrator)
    run([
        sys.executable,
        "-m", "src.eval.ml_roi_analysis",
        "--per-game", str(PER_GAME_FIXTURE),
        "--mode", "ev_cal",
        "--delta-calibrator", str(delta_cal),
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--max-bet-rate", "0.30",
    ])


def run_commit2_historical_and_backtest_if_enabled() -> None:
    """
    Optional commit-2 verification:
      - historical_prediction_runner produces predictions_YYYY-MM-DD.csv + historical audit
      - backtest joins predictions to results + backtest audit

    Controlled by environment variables so CI doesn’t require your large history file.
    """
    enable = os.environ.get("NBA_CERTIFY_COMMIT2", "").strip().lower() in {"1", "true", "yes"}
    if not enable:
        print("\n[certify] (commit2) Skipping historical/backtest steps (set NBA_CERTIFY_COMMIT2=1 to enable).")
        return

    history_csv = os.environ.get("NBA_HISTORY_CSV", "").strip()
    results_csv = os.environ.get("NBA_RESULTS_CSV", "").strip()
    pred_dir = os.environ.get("NBA_PRED_DIR", "outputs").strip()

    if not history_csv or not results_csv:
        raise RuntimeError(
            "[certify] (commit2) NBA_CERTIFY_COMMIT2=1 but NBA_HISTORY_CSV and/or NBA_RESULTS_CSV not set."
        )

    print("\n[certify] === Commit 2: Historical Runner ===")
    run([
        sys.executable,
        "-m", "src.eval.historical_prediction_runner",
        "--history", history_csv,
        "--start", TRAIN_START,
        "--end", EVAL_END,
        "--apply-market",
        "--overwrite",
    ])

    if not HISTORICAL_AUDIT.exists():
        raise RuntimeError(f"[certify] Missing expected audit: {HISTORICAL_AUDIT}")

    print("\n[certify] === Commit 2: Backtest Join ===")
    run([
        sys.executable,
        "-m", "src.eval.backtest",
        "--pred-dir", pred_dir,
        "--results", results_csv,
        "--start", TRAIN_START,
        "--end", EVAL_END,
        "--strict",
    ])

    if not BACKTEST_AUDIT.exists():
        raise RuntimeError(f"[certify] Missing expected audit: {BACKTEST_AUDIT}")

    print(f"[certify] ✔ {HISTORICAL_AUDIT.name}")
    print(f"[certify] ✔ {BACKTEST_AUDIT.name}")


def verify_outputs() -> None:
    print("\n[certify] === Verifying Outputs ===")
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        raise RuntimeError(
            "[certify] Missing required outputs:\n" + "\n".join(str(p) for p in missing)
        )

    for p in REQUIRED_OUTPUTS:
        print(f"[certify] ✔ {p.name}")


def main() -> None:
    print("\n[certify] Starting NBA Certification Suite")
    print("[certify] Repo:", REPO_ROOT)

    ensure_paths()
    run_ats_regression()
    run_totals_suite()
    run_ml_suite()

    # commit 2 (optional)
    run_commit2_historical_and_backtest_if_enabled()

    verify_outputs()

    print("\n[certify] ✅ CERTIFICATION PASSED")
    print("[certify] System is reproducible and healthy.")


if __name__ == "__main__":
    main()
