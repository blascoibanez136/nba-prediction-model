from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
SNAPSHOT_DIR = REPO_ROOT / "data" / "_snapshots"


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _info(msg: str) -> None:
    print(f"[historical] {msg}")


def _warn(msg: str) -> None:
    print(f"[historical][WARNING] {msg}")


def _find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_pickle(path: Path) -> Any:
    import joblib  # local import to avoid import-time issues

    return joblib.load(path)


def _load_teams(path: Path) -> Any:
    """
    Teams artifact can be either:
    - JSON (preferred for portability): team_index.json
    - Pickle/joblib legacy: teams.pkl / team_index.pkl / etc.
    """
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return _load_pickle(path)


def _locate_model_artifacts() -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """
    Locate model artifacts without assuming a fixed directory exists.

    Commit-2 fix: include REPO_ROOT/models (Colab upload target) as a first-class base.

    Returns:
      (paths, attempted_payload)
    """
    bases = [
        REPO_ROOT / "models",  # ✅ commit-2 fix: primary location for Colab uploads
        REPO_ROOT / "artifacts",
        REPO_ROOT / "artifacts" / "models",
        REPO_ROOT / "outputs",
        REPO_ROOT,
    ]

    # conservative, allow multiple conventions
    win_names = ["win_model.pkl", "win_model.joblib", "model_win.pkl"]
    spread_names = ["spread_model.pkl", "spread_model.joblib", "model_spread.pkl"]
    total_names = ["total_model.pkl", "total_model.joblib", "model_total.pkl"]

    # ✅ commit-2 fix: support JSON team index as canonical
    teams_names = [
        "team_index.json",
        "teams.json",
        "teams.pkl",
        "teams.joblib",
        "team_index.pkl",
        "teams_map.pkl",
    ]

    def expand(names: List[str]) -> List[Path]:
        out: List[Path] = []
        for b in bases:
            for n in names:
                out.append(b / n)
        return out

    win_candidates = expand(win_names)
    spread_candidates = expand(spread_names)
    total_candidates = expand(total_names)
    teams_candidates = expand(teams_names)

    win_path = _find_first_existing(win_candidates)
    spread_path = _find_first_existing(spread_candidates)
    total_path = _find_first_existing(total_candidates)
    teams_path = _find_first_existing(teams_candidates)

    attempted = {
        "bases": [str(b) for b in bases],
        "win_candidates": [str(p) for p in win_candidates],
        "spread_candidates": [str(p) for p in spread_candidates],
        "total_candidates": [str(p) for p in total_candidates],
        "teams_candidates": [str(p) for p in teams_candidates],
    }

    missing: List[str] = []
    if win_path is None:
        missing.append("win_model")
    if spread_path is None:
        missing.append("spread_model")
    if total_path is None:
        missing.append("total_model")
    if teams_path is None:
        missing.append("teams")

    if missing:
        raise RuntimeError(
            "Could not locate required model artifacts: "
            + ", ".join(missing)
            + ". Check attempted paths in audit."
        ) from RuntimeError(json.dumps({"missing": missing, "attempted": attempted}))

    return (
        {
            "teams": teams_path,
            "win_model": win_path,
            "spread_model": spread_path,
            "total_model": total_path,
        },
        attempted,
    )


def _load_models(audit: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    paths, attempted = _locate_model_artifacts()
    audit["model_artifacts"] = {k: str(v) for k, v in paths.items()}
    audit["artifact_search_attempted"] = attempted

    teams = _load_teams(paths["teams"])
    win_model = _load_pickle(paths["win_model"])
    spread_model = _load_pickle(paths["spread_model"])
    total_model = _load_pickle(paths["total_model"])
    return teams, win_model, spread_model, total_model


def _run_predict_for_day(
    df_day: pd.DataFrame,
    teams: Any,
    win_model: Any,
    spread_model: Any,
    total_model: Any,
) -> pd.DataFrame:
    from src.model.predict import predict_games  # canonical in your repo

    return predict_games(
        df_day,
        teams=teams,
        win_model=win_model,
        spread_model=spread_model,
        total_model=total_model,
    )


def _try_apply_market(df_pred: pd.DataFrame, game_date: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diag: Dict[str, Any] = {"attempted": True, "applied": False, "reason": None}

    snap = SNAPSHOT_DIR / f"odds_snapshot_close_{game_date}.csv"
    if not snap.exists():
        diag["reason"] = f"no_close_snapshot:{snap.name}"
        return df_pred, diag

    try:
        from src.model.market_ensemble import apply_market_ensemble  # type: ignore

        df_out = apply_market_ensemble(df_pred, snapshot_path=str(snap))
        diag["applied"] = True
        return df_out, diag
    except Exception as e:
        diag["reason"] = f"market_apply_error:{type(e).__name__}"
        return df_pred, diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Historical prediction runner with audit output.")
    ap.add_argument("--history", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--apply-market", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    audit: Dict[str, Any] = {
        "status": "ok",
        "window": {"start": args.start, "end": args.end},
        "inputs": {
            "history": args.history,
            "apply_market": bool(args.apply_market),
            "overwrite": bool(args.overwrite),
        },
        "model_artifacts": {},
        "artifact_search_attempted": {},
        "days": {"processed": 0, "written": 0, "skipped_existing": 0, "skipped_no_games": 0},
        "market": {"attempted_days": 0, "applied_days": 0, "skipped_days": 0, "skip_reasons": {}},
        "outputs_sample": [],
        "errors": [],
    }

    audit_path = OUTPUTS_DIR / "historical_prediction_runner_audit.json"

    try:
        hist = pd.read_csv(args.history)
        if "game_date" not in hist.columns:
            raise RuntimeError("history CSV missing required column: game_date")
        hist["game_date"] = hist["game_date"].astype(str)

        teams, win_model, spread_model, total_model = _load_models(audit)
        _info("loaded models (teams + win/spread/total)")

        start_dt = _parse_date(args.start)
        end_dt = _parse_date(args.end)

        d = start_dt
        while d <= end_dt:
            game_date = _date_str(d)
            audit["days"]["processed"] += 1

            out_path = OUTPUTS_DIR / f"predictions_{game_date}.csv"
            if out_path.exists() and not args.overwrite:
                audit["days"]["skipped_existing"] += 1
                d += timedelta(days=1)
                continue

            df_day = hist[hist["game_date"] == game_date].copy()
            if df_day.empty:
                audit["days"]["skipped_no_games"] += 1
                d += timedelta(days=1)
                continue

            df_pred = _run_predict_for_day(df_day, teams, win_model, spread_model, total_model)

            if args.apply_market:
                audit["market"]["attempted_days"] += 1
                df_pred2, diag = _try_apply_market(df_pred, game_date)
                if diag.get("applied"):
                    audit["market"]["applied_days"] += 1
                    df_pred = df_pred2
                else:
                    audit["market"]["skipped_days"] += 1
                    r = diag.get("reason") or "unknown"
                    audit["market"]["skip_reasons"][r] = int(audit["market"]["skip_reasons"].get(r, 0) + 1)
                    if r.startswith("no_close_snapshot"):
                        _warn(f"No CLOSE odds snapshot found for {game_date} in data/_snapshots — skipping market ensemble")
                    else:
                        _warn(f"Market ensemble skipped for {game_date}: {r}")

            df_pred.to_csv(out_path, index=False)
            audit["days"]["written"] += 1
            if len(audit["outputs_sample"]) < 15:
                audit["outputs_sample"].append({"date": game_date, "path": str(out_path), "rows": int(len(df_pred))})
            _info(f"Wrote {out_path} ({len(df_pred)} rows)")

            d += timedelta(days=1)

    except Exception as e:
        audit["status"] = "fail"
        audit["errors"].append({"type": type(e).__name__, "message": str(e)})
        if e.__cause__ is not None and isinstance(e.__cause__, RuntimeError):
            try:
                payload = json.loads(str(e.__cause__))
                audit["artifact_search_failure"] = payload
            except Exception:
                pass
        raise
    finally:
        _write_json(audit_path, audit)
        _info("wrote outputs/historical_prediction_runner_audit.json")


if __name__ == "__main__":
    main()
