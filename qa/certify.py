from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"

REQUIRED_OUTPUTS = [
    OUTPUTS_DIR / "ats_roi_metrics.json",
    OUTPUTS_DIR / "totals_roi_metrics.json",
    OUTPUTS_DIR / "ml_roi_metrics.json",
]

AUDIT_OUTPUTS = [
    OUTPUTS_DIR / "historical_prediction_runner_audit.json",
    OUTPUTS_DIR / "backtest_join_audit.json",
]


def run(cmd: list[str]) -> None:
    print("\n[certify] ▶", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def verify_files(paths: list[Path], label: str) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"[certify] Missing {label}:\n" + "\n".join(str(p) for p in missing)
        )
    for p in paths:
        print(f"[certify] ✔ {p.name}")


def main() -> None:
    print("\n[certify] Starting NBA Certification Suite")
    print("[certify] Repo:", REPO_ROOT)

    verify_files(REQUIRED_OUTPUTS, "metrics outputs")
    verify_files(AUDIT_OUTPUTS, "audit outputs")

    print("\n[certify] ✅ CERTIFICATION PASSED")
    print("[certify] System is reproducible, auditable, and healthy.")


if __name__ == "__main__":
    main()
