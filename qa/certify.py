from __future__ import annotations

import argparse
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

# Commit 2 audit outputs (only required if corresponding step runs)
BACKTEST_AUDIT = OUTPUTS_DIR / "backtest_join_audit.json"
HISTORICAL_AUDIT = OUTPUTS_DIR / "historical_prediction_runner_audit.json"


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


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[certify] Expected {label} at {path} but it does not exist.")


def run_backtest_step(results_path: str, pred_dir: str = "outputs") -> None:
    """
    Runs src.eval.backtest to produce backtest metrics and join audit JSON.
    """
    print("\n[certify] === Backtest (optional) ===")
    run([
        sys.executable,
        "-m", "src.eval.backtest",
        "--pred-dir", pred_dir,
        "--pattern", "predictions_*_market.csv",
        "--results", results_path,
        "--start", TRAIN_START,
        "--end", EVAL_END,
        "--metrics-path", str(OUTPUTS_DIR / "backtest_metrics.json"),
        "--calib-path", str(OUTPUTS_DIR / "backtest_calibration.csv"),
        "--per-game-path", str(OUTPUTS_DIR / "backtest_per_game.csv"),
    ])

    # ✅ fail-fast: confirm audit exists immediately after step
    _require_file(BACKTEST_AUDIT, "backtest join audit")


def run_historical_step(history_path: str, apply_market: bool, overwrite: bool) -> None:
    """
    Runs src.eval.historical_prediction_runner and expects it to write an audit JSON.
    """
    print("\n[certify] === Historical Prediction Runner (optional) ===")
    cmd = [
        sys.executable,
        "-m", "src.eval.historical_prediction_runner",
        "--history", history_path,
        "--start", TRAIN_START,
        "--end", EVAL_END,
    ]
    if apply_market:
        cmd.append("--apply-market")
    if overwrite:
        cmd.append("--overwrite")
    run(cmd)

    # ✅ fail-fast: confirm audit exists immediately after step
    _require_file(HISTORICAL_AUDIT, "historical runner audit")


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


def verify_optional_audits(require_backtest_audit: bool, require_historical_audit: bool) -> None:
    print("\n[certify] === Verifying Audit Outputs ===")
    missing = []
    if require_backtest_audit and not BACKTEST_AUDIT.exists():
        missing.append(BACKTEST_AUDIT)
    if require_historical_audit and not HISTORICAL_AUDIT.exists():
        missing.append(HISTORICAL_AUDIT)

    if missing:
        raise RuntimeError(
            "[certify] Missing required audit outputs:\n"
            + "\n".join(str(p) for p in missing)
        )

    if require_backtest_audit:
        print(f"[certify] ✔ {BACKTEST_AUDIT.name}")
    if require_historical_audit:
        print(f"[certify] ✔ {HISTORICAL_AUDIT.name}")


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


def write_run_metadata(ran_backtest: bool, ran_historical: bool) -> None:
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
        "steps": {
            "ats_regression": True,
            "totals_suite": True,
            "ml_suite": True,
            "backtest_step": bool(ran_backtest),
            "historical_step": bool(ran_historical),
        },
        "inputs": {"per_game_fixture": _safe_stat(PER_GAME_FIXTURE)},
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
            "backtest_join_audit": _safe_stat(BACKTEST_AUDIT),
            "historical_runner_audit": _safe_stat(HISTORICAL_AUDIT),
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run NBA Certification Suite (ATS/Totals/ML) + optional audits.")
    ap.add_argument("--run-backtest", action="store_true")
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--run-historical", action="store_true")
    ap.add_argument("--history-path", default=None)
    ap.add_argument("--historical-apply-market", action="store_true")
    ap.add_argument("--historical-overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print("\n[certify] Starting NBA Certification Suite")
    print("[certify] Repo:", REPO_ROOT)

    ensure_paths()

    # Core locked suite
    run_ats_regression()
    run_totals_suite()
    run_ml_suite()
    verify_outputs()

    ran_backtest = False
    ran_historical = False

    if args.run_backtest:
        if not args.results_path:
            raise RuntimeError("[certify] --run-backtest requires --results-path")
        run_backtest_step(results_path=args.results_path)
        ran_backtest = True

    if args.run_historical:
        if not args.history_path:
            raise RuntimeError("[certify] --run-historical requires --history-path")
        run_historical_step(
            history_path=args.history_path,
            apply_market=bool(args.historical_apply_market),
            overwrite=bool(args.historical_overwrite),
        )
        ran_historical = True

    verify_optional_audits(require_backtest_audit=ran_backtest, require_historical_audit=ran_historical)
    write_run_metadata(ran_backtest=ran_backtest, ran_historical=ran_historical)

    print("\n[certify] ✅ CERTIFICATION PASSED")
    print("[certify] System is reproducible and healthy.")


if __name__ == "__main__":
    main()
