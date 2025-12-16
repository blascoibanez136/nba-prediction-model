from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict


EXPECTED_POLICY_HASH = "1e2785d7939cac32b605dd0b35cd91cda8c41dec8fb8b58b42d7b2e303ab2258"

# System-of-record windows (LOCKED)
TRAIN_START = "2023-10-24"
TRAIN_END = "2024-03-10"
EVAL_START = "2024-03-11"
EVAL_END = "2024-04-14"

# Paths (repo-relative)
FIXTURE_PER_GAME = "qa/fixtures/backtest_per_game.csv"
POLICY_PATH = "configs/ats_policy_v1.yaml"

# Conservative regression bounds (avoid flakiness but catch breakage)
BET_RATE_MIN = 0.08
BET_RATE_MAX = 0.30
BETS_MIN = 30
BETS_MAX = 120
ROI_MIN = 0.05  # not your best-case; just ensures edge didn't collapse
WIN_RATE_MIN = 0.52


def _run(cmd: list[str], env: Dict[str, str] | None = None) -> None:
    print(f"[qa][run] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    repo_root = os.getcwd()
    per_game = os.path.join(repo_root, FIXTURE_PER_GAME)
    policy = os.path.join(repo_root, POLICY_PATH)

    if not os.path.exists(per_game):
        raise RuntimeError(
            f"[qa] Missing fixture: {FIXTURE_PER_GAME}\n"
            f"Create it from a known-good backtest_per_game.csv and commit it.\n"
            f"Expected columns include game_date, home/away scores, home_spread_consensus, fair_spread_model, "
            f"home_spread_dispersion, etc."
        )
    if not os.path.exists(policy):
        raise RuntimeError(f"[qa] Missing policy file: {POLICY_PATH}")

    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root  # ensure `src.*` imports resolve

    with tempfile.TemporaryDirectory() as td:
        artifacts_dir = os.path.join(td, "artifacts")
        outputs_dir = os.path.join(td, "outputs")
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

        calibrator_path = os.path.join(artifacts_dir, "spread_calibrator.joblib")

        # 1) Train calibrator on LOCKED train window
        _run(
            [
                sys.executable,
                "-m",
                "src.eval.train_spread_calibrator",
                "--per-game",
                per_game,
                "--train-start",
                TRAIN_START,
                "--train-end",
                TRAIN_END,
                "--out",
                calibrator_path,
            ],
            env=env,
        )

        # 2) Run ATS ROI analysis on LOCKED eval window w/ strict + hash enforcement
        _run(
            [
                sys.executable,
                "-m",
                "src.eval.ats_roi_analysis",
                "--per_game",
                per_game,
                "--calibrator",
                calibrator_path,
                "--policy",
                policy,
                "--require-policy-hash",
                EXPECTED_POLICY_HASH,
                "--eval-start",
                EVAL_START,
                "--eval-end",
                EVAL_END,
                "--strict",
            ],
            env=env,
        )

        # NOTE: ats_roi_analysis writes to repo-root outputs/ by default.
        # We deliberately validate that output as the system-of-record artifact.
        metrics_path = os.path.join(repo_root, "outputs", "ats_roi_metrics.json")
        if not os.path.exists(metrics_path):
            raise RuntimeError("[qa] ats_roi_metrics.json not found after ATS analysis run.")

        metrics = _load_json(metrics_path)

        # 3) Hard assertions (regression gates)
        policy_block = metrics.get("policy", {})
        got_hash = str(policy_block.get("hash", "")).strip()
        if got_hash != EXPECTED_POLICY_HASH:
            raise RuntimeError(f"[qa] POLICY HASH REGRESSION: expected={EXPECTED_POLICY_HASH} got={got_hash}")

        bet_rate = float(metrics.get("bet_rate", -1))
        if not (BET_RATE_MIN <= bet_rate <= BET_RATE_MAX):
            raise RuntimeError(f"[qa] BET RATE OUT OF BOUNDS: bet_rate={bet_rate:.3f} expected {BET_RATE_MIN}..{BET_RATE_MAX}")

        overall = metrics.get("overall", {})
        bets = int(overall.get("bets", -1))
        roi = overall.get("roi", None)
        win_rate = overall.get("win_rate", None)

        if not (BETS_MIN <= bets <= BETS_MAX):
            raise RuntimeError(f"[qa] BET COUNT OUT OF BOUNDS: bets={bets} expected {BETS_MIN}..{BETS_MAX}")

        if roi is None:
            raise RuntimeError("[qa] ROI missing from metrics.overall.roi")
        roi = float(roi)
        if roi < ROI_MIN:
            raise RuntimeError(f"[qa] ROI COLLAPSE: roi={roi:.4f} < {ROI_MIN:.4f}")

        if win_rate is None:
            raise RuntimeError("[qa] win_rate missing from metrics.overall.win_rate")
        win_rate = float(win_rate)
        if win_rate < WIN_RATE_MIN:
            raise RuntimeError(f"[qa] WIN RATE COLLAPSE: win_rate={win_rate:.4f} < {WIN_RATE_MIN:.4f}")

        # 4) Leak guard: ensure eval window is exactly the locked range
        eval_window = metrics.get("eval_window", {})
        if str(eval_window.get("start")) != EVAL_START or str(eval_window.get("end")) != EVAL_END:
            raise RuntimeError(
                f"[qa] EVAL WINDOW REGRESSION: expected {EVAL_START}..{EVAL_END} "
                f"got {eval_window.get('start')}..{eval_window.get('end')}"
            )

        print("[qa] ✅ ATS Policy v1 regression gate PASSED.")
        print(f"[qa] bets={bets} bet_rate={bet_rate:.3f} roi={roi:.4f} win_rate={win_rate:.4f} policy_hash={got_hash}")


if __name__ == "__main__":
    main()
