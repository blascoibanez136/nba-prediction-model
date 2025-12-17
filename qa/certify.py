from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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


# -------------------------
# subprocess helpers
# -------------------------


def run(cmd: list[str]) -> None:
    print("\n[certify] ▶", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def ensure_paths() -> None:
    if not PER_GAME_FIXTURE.exists():
        raise RuntimeError(f"[certify] Missing fixture: {PER_GAME_FIXTURE}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


# -------------------------
# certification suites
# -------------------------


def run_ats_regression() -> None:
    print("\n[certify] === ATS Policy v1 Regression ===")
    run([sys.executable, "qa/regression_ats_policy_v1.py"])


def run_totals_suite() -> None:
    """
    Totals v3 notes:
    - train_total_calibrator: expects --per-game (dash)
    - totals_roi_analysis: expects --per-game (dash) and does NOT accept --pricing/--strict
    """
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
    """
    ML Selector v3 notes:
    - train_delta_calibrator: expects --per_game (underscore)
    - ml_roi_analysis: expects --per_game (underscore)
    - delta calibrator bundle must be passed via --delta-calibrator (NOT --calibrator)
    """
    print("\n[certify] === ML Selector v3 ===")

    delta_cal = ARTIFACTS_DIR / "delta_calibrator.joblib"

    run([
        sys.executable,
        "-m", "src.eval.train_delta_calibrator",
        "--per_game", str(PER_GAME_FIXTURE),
        "--train-start", TRAIN_START,
        "--train-end", TRAIN_END,
        "--out", str(delta_cal),
    ])

    run([
        sys.executable,
        "-m", "src.eval.ml_roi_analysis",
        "--per_game", str(PER_GAME_FIXTURE),
        "--mode", "ev_cal",
        "--delta-calibrator", str(delta_cal),
        "--eval-start", EVAL_START,
        "--eval-end", EVAL_END,
        "--max-bet-rate", "0.30",
    ])


# -------------------------
# outputs + metadata
# -------------------------


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_stat(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": st.st_size,
        "sha256": _sha256_file(path),
    }


def _get_git_commit(repo_root: Path) -> Optional[str]:
    env_sha = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _try_load_overall(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        overall = data.get("overall", {})
        return overall if isinstance(overall, dict) else {}
    except Exception:
        return {}


def write_run_metadata() -> None:
    """
    Deterministic run receipt for certification runs.
    Additive only: does not affect selectors/policies/results.
    """
    meta: Dict[str, Any] = {
        "kind": "certification_suite",
        "generated_at_utc": _utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "git_commit": _get_git_commit(REPO_ROOT),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "windows": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "eval_start": EVAL_START,
            "eval_end": EVAL_END,
        },
        "inputs": {
            "per_game_fixture": _safe_stat(PER_GAME_FIXTURE),
        },
        "artifacts": {
            "total_calibrator": _safe_stat(ARTIFACTS_DIR / "total_calibrator.joblib"),
            "delta_calibrator": _safe_stat(ARTIFACTS_DIR / "delta_calibrator.joblib"),
        },
        "outputs": {
            "ats_metrics": _safe_stat(OUTPUTS_DIR / "ats_roi_metrics.json"),
            "totals_metrics": _safe_stat(OUTPUTS_DIR / "totals_roi_metrics.json"),
            "ml_metrics": _safe_stat(OUTPUTS_DIR / "ml_roi_metrics.json"),
            "ats_bets": _safe_stat(OUTPUTS_DIR / "ats_roi_bets.csv"),
            "totals_bets": _safe_stat(OUTPUTS_DIR / "totals_roi_bets.csv"),
            "ml_bets": _safe_stat(OUTPUTS_DIR / "ml_roi_bets.csv"),
        },
        "summary": {
            "ats_overall": _try_load_overall(OUTPUTS_DIR / "ats_roi_metrics.json"),
            "totals_overall": _try_load_overall(OUTPUTS_DIR / "totals_roi_metrics.json"),
            "ml_overall": _try_load_overall(OUTPUTS_DIR / "ml_roi_metrics.json"),
        },
    }

    out_path = OUTPUTS_DIR / "run_metadata.json"
    _write_json(out_path, meta)
    print("[certify] wrote: outputs/run_metadata.json")


def main() -> None:
    print("\n[certify] Starting NBA Certification Suite")
    print("[certify] Repo:", REPO_ROOT)

    ensure_paths()
    run_ats_regression()
    run_totals_suite()
    run_ml_suite()
    verify_outputs()
    write_run_metadata()

    print("\n[certify] ✅ CERTIFICATION PASSED")
    print("[certify] System is reproducible and healthy.")


if __name__ == "__main__":
    main()
