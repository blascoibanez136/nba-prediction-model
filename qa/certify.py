from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def run(cmd: list[str]) -> None:
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
        "--per_game", str(PER_GAME_FIXTURE),
        "--calibrator", str(total_cal),
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--pricing", "fixed_-110",
        "--max-bet-rate", "0.30",
        "--strict",
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

    run([
        sys.executable,
        "-m", "src.eval.ml_roi_analysis",
        "--per_game", str(PER_GAME_FIXTURE),
        "--mode", "ev_cal",
        "--calibrator", str(delta_cal),
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--max-bet-rate", "0.30",
        "--strict",
    ])


def verify_outputs() -> None:
    print("\n[certify] === Verifying Outputs ===")
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        raise RuntimeError(
            "[certify] Missing required outputs:\n"
            + "\n".join(str(p) for p in missing)
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
    verify_outputs()

    print("\n[certify] ✅ CERTIFICATION PASSED")
    print("[certify] System is reproducible and healthy.")


if __name__ == "__main__":
    main()
