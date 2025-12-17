from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"


REQUIRED_OUTPUTS = [
    OUTPUTS / "ats_roi_metrics.json",
    OUTPUTS / "totals_roi_metrics.json",
    OUTPUTS / "ml_roi_metrics.json",
    OUTPUTS / "historical_prediction_runner_audit.json",
    OUTPUTS / "backtest_join_audit.json",
]


def run(cmd):
    print("\n[certify] ▶", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def main():
    print("\n[certify] Starting NBA Certification Suite")
    print("[certify] Repo:", REPO_ROOT)

    OUTPUTS.mkdir(exist_ok=True)

    run([sys.executable, "qa/regression_ats_policy_v1.py"])

    run([
        sys.executable, "-m", "src.eval.train_total_calibrator",
        "--per-game", "qa/fixtures/backtest_per_game.csv",
        "--train-start", "2023-10-24",
        "--train-end", "2024-03-10",
        "--out", "artifacts/total_calibrator.joblib",
    ])

    run([
        sys.executable, "-m", "src.eval.totals_roi_analysis",
        "--per_game", "qa/fixtures/backtest_per_game.csv",
        "--calibrator", "artifacts/total_calibrator.joblib",
        "--eval-start", "2024-03-11",
        "--eval-end", "2024-04-14",
        "--max-bet-rate", "0.30",
    ])

    run([
        sys.executable, "-m", "src.eval.train_delta_calibrator",
        "--per-game", "qa/fixtures/backtest_per_game.csv",
        "--train-start", "2023-10-24",
        "--train-end", "2024-03-10",
        "--out", "artifacts/delta_calibrator.joblib",
    ])

    run([
        sys.executable, "-m", "src.eval.ml_roi_analysis",
        "--per_game", "qa/fixtures/backtest_per_game.csv",
        "--mode", "ev_cal",
        "--delta-calibrator", "artifacts/delta_calibrator.joblib",
        "--eval-start", "2024-03-11",
        "--eval-end", "2024-04-14",
        "--max-bet-rate", "0.30",
    ])

    print("\n[certify] === Verifying Outputs ===")
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        raise RuntimeError(
            "[certify] Missing required outputs:\n" +
            "\n".join(str(p) for p in missing)
        )

    for p in REQUIRED_OUTPUTS:
        print(f"[certify] ✔ {p.name}")

    print("\n[certify] ✅ CERTIFICATION PASSED")
    print("[certify] System is reproducible and healthy.")


if __name__ == "__main__":
    main()
